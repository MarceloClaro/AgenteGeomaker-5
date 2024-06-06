import streamlit as st
import base64
import json
import os
import datetime
import requests
from bs4 import BeautifulSoup
from collections import namedtuple
import io
from PIL import Image
import re
import configparser
import tenacity
import tiktoken
from typing import Tuple
from groq import Groq

# Configuração da página
st.set_page_config(layout="wide")

# Definição de variáveis globais e classes
FILEPATH = "agents.json"
MODEL_MAX_TOKENS = {
    'mixtral-8x7b-32768': 32768,
    'llama3-70b-8192': 8192,
    'llama3-8b-8192': 8192,
    'gemma-7b-it': 8192,
}

# Verificação e criação do diretório necessário
STATIC_DIRECTORY = 'static'
if not os.path.exists(STATIC_DIRECTORY):
    os.makedirs(STATIC_DIRECTORY)

import fitz  # PyMuPDF

ArxivParams = namedtuple(
    "ArxivParams",
    ["query", "key_word", "page_num", "max_results", "days", "sort", "save_image", "file_format", "language"],
)

class Paper:
    def __init__(self, path, title='', url='', abs='', authers=[]):
        self.url = url
        self.path = path
        self.section_names = []
        self.section_texts = {}
        self.abs = abs
        self.title_page = 0
        self.title = title
        self.pdf = fitz.open(self.path)
        self.parse_pdf()
        self.authers = authers
        self.roman_num = ["I", "II", 'III', "IV", "V", "VI", "VII", "VIII", "IIX", "IX", "X"]
        self.digit_num = [str(d + 1) for d in range(10)]
        self.first_image = ''

    def parse_pdf(self):
        self.pdf = fitz.open(self.path)
        self.text_list = [page.get_text() for page in self.pdf]
        self.all_text = ' '.join(self.text_list)
        self.section_page_dict = self._get_all_page_index()
        self.section_text_dict = self._get_all_page()
        self.section_text_dict.update({"title": self.title})
        self.section_text_dict.update({"paper_info": self.get_paper_info()})
        self.pdf.close()

    def get_paper_info(self):
        first_page_text = self.pdf[self.title_page].get_text()
        if "Abstract" in self.section_text_dict.keys():
            abstract_text = self.section_text_dict['Abstract']
        else:
            abstract_text = self.abs
        first_page_text = first_page_text.replace(abstract_text, "")
        return first_page_text

    def get_image_path(self, image_path=''):
        max_size = 0
        image_list = []
        with fitz.Document(self.path) as my_pdf_file:
            for page_number in range(1, len(my_pdf_file) + 1):
                page = my_pdf_file[page_number - 1]
                images = page.get_images()
                for image_number, image in enumerate(page.get_images(), start=1):
                    xref_value = image[0]
                    base_image = my_pdf_file.extract_image(xref_value)
                    image_bytes = base_image["image"]
                    ext = base_image["ext"]
                    image = Image.open(io.BytesIO(image_bytes))
                    image_size = image.size[0] * image.size[1]
                    if image_size > max_size:
                        max_size = image_size
                    image_list.append(image)
        for image in image_list:
            image_size = image.size[0] * image.size[1]
            if image_size == max_size:
                image_name = f"image.{ext}"
                im_path = os.path.join(image_path, image_name)
                max_pix = 480
                origin_min_pix = min(image.size[0], image.size[1])
                if image.size[0] > image.size[1]:
                    min_pix = int(image.size[1] * (max_pix / image.size[0]))
                    newsize = (max_pix, min_pix)
                else:
                    min_pix = int(image.size[0] * (max_pix / image.size[1]))
                    newsize = (min_pix, max_pix)
                image = image.resize(newsize)
                image.save(open(im_path, "wb"))
                return im_path, ext
        return None, None

    def get_chapter_names(self):
        doc = fitz.open(self.path)
        text_list = [page.get_text() for page in doc]
        all_text = ''
        for text in text_list:
            all_text += text
        chapter_names = []
        for line in all_text.split('\n'):
            line_list = line.split(' ')
            if '.' in line:
                point_split_list = line.split('.')
                space_split_list = line.split(' ')
                if 1 < len(space_split_list) < 5:
                    if 1 < len(point_split_list) < 5 and (
                            point_split_list[0] in self.roman_num or point_split_list[0] in self.digit_num):
                        chapter_names.append(line)
                    elif 1 < len(point_split_list) < 5:
                        chapter_names.append(line)
        return chapter_names

    def get_title(self):
        doc = self.pdf
        max_font_size = 0
        max_string = ""
        max_font_sizes = [0]
        for page_index, page in enumerate(doc):
            text = page.get_text("dict")
            blocks = text["blocks"]
            for block in blocks:
                if block["type"] == 0 and len(block['lines']):
                    if len(block["lines"][0]["spans"]):
                        font_size = block["lines"][0]["spans"][0]["size"]
                        max_font_sizes.append(font_size)
                        if font_size > max_font_size:
                            max_font_size = font_size
                            max_string = block["lines"][0]["spans"][0]["text"]
        max_font_sizes.sort()
        cur_title = ''
        for page_index, page in enumerate(doc):
            text = page.get_text("dict")
            blocks = text["blocks"]
            for block in blocks:
                if block["type"] == 0 and len(block['lines']):
                    if len(block["lines"][0]["spans"]):
                        cur_string = block["lines"][0]["spans"][0]["text"]
                        font_flags = block["lines"][0]["spans"][0]["flags"]
                        font_size = block["lines"][0]["spans"][0]["size"]
                        if abs(font_size - max_font_sizes[-1]) < 0.3 or abs(font_size - max_font_sizes[-2]) < 0.3:
                            if len(cur_string) > 4 and "arXiv" not in cur_string:
                                if cur_title == '':
                                    cur_title += cur_string
                                else:
                                    cur_title += ' ' + cur_string
                            self.title_page = page_index
        title = cur_title.replace('\n', ' ')
        return title

    def _get_all_page_index(self):
        section_list = ["Abstract",
                        'Introduction', 'Related Work', 'Background',
                        "Introduction and Motivation", "Computation Function", "Routing Function",
                        "Preliminary", "Problem Formulation",
                        'Methods', 'Methodology', "Method", 'Approach', 'Approaches',
                        "Materials and Methods", "Experiment Settings",
                        'Experiment', "Experimental Results", "Evaluation", "Experiments",
                        "Results", 'Findings', 'Data Analysis',
                        "Discussion", "Results and Discussion", "Conclusion",
                        'References']
        section_page_dict = {}
        for page_index, page in enumerate(self.pdf):
            cur_text = page.get_text()
            for section_name in section_list:
                section_name_upper = section_name.upper()
                if "Abstract" == section_name and section_name in cur_text:
                    section_page_dict[section_name] = page_index
                else:
                    if section_name + '\n' in cur_text:
                        section_page_dict[section_name] = page_index
                    elif section_name_upper + '\n' in cur_text:
                        section_page_dict[section_name] = page_index
        return section_page_dict

    def _get_all_page(self):
        text = ''
        text_list = []
        section_dict = {}
        text_list = [page.get_text() for page in self.pdf]
        for sec_index, sec_name in enumerate(self.section_page_dict):
            if sec_index <= 0 and self.abs:
                continue
            else:
                start_page = self.section_page_dict[sec_name]
                if sec_index < len(list(self.section_page_dict.keys())) - 1:
                    end_page = self.section_page_dict[list(self.section_page_dict.keys())[sec_index + 1]]
                else:
                    end_page = len(text_list)
                cur_sec_text = ''
                if start_page == end_page:
                    if sec_index < len(list(self.section_page_dict.keys())) - 1:
                        next_sec = list(self.section_page_dict.keys())[sec_index + 1]
                        if text_list[start_page].find(sec_name) == -1:
                            start_i = text_list[start_page].find(sec_name.upper())
                        else:
                            start_i = text_list[start_page].find(sec_name)
                        if text_list[start_page].find(next_sec) == -1:
                            end_i = text_list[start_page].find(next_sec.upper())
                        else:
                            end_i = text_list[start_page].find(next_sec)
                        cur_sec_text += text_list[start_page][start_i:end_i]
                else:
                    for page_i in range(start_page, end_page):
                        if page_i == start_page:
                            if text_list[start_page].find(sec_name) == -1:
                                start_i = text_list[start_page].find(sec_name.upper())
                            else:
                                start_i = text_list[start_page].find(sec_name)
                            cur_sec_text += text_list[page_i][start_i:]
                        elif page_i < end_page:
                            cur_sec_text += text_list[page_i]
                        elif page_i == end_page:
                            if sec_index < len(list(self.section_page_dict.keys())) - 1:
                                next_sec = list(self.section_page_dict.keys())[sec_index + 1]
                                if text_list[start_page].find(next_sec) == -1:
                                    end_i = text_list[start_page].find(next_sec.upper())
                                else:
                                    end_i = text_list[start_page].find(next_sec)
                                cur_sec_text += text_list[page_i][:end_i]
                section_dict[sec_name] = cur_sec_text.replace('-\n', '').replace('\n', ' ')
        return section_dict

class Reader:
    def __init__(self, key_word, query, root_path='./', gitee_key='', sort=None, user_name='defualt', args=None):
        self.user_name = user_name
        self.key_word = key_word
        self.query = query
        self.sort = sort
        self.args = args
        if args.language == 'en':
            self.language = 'English'
        elif args.language == 'zh':
            self.language = 'Chinese'
        else:
            self.language = 'Chinese'
        self.root_path = root_path
        self.config = configparser.ConfigParser()
        self.config.read('apikey.ini')
        OPENAI_KEY = os.environ.get("OPENAI_KEY", "")
        self.chat_api_list = self.config.get('OpenAI', 'OPENAI_API_KEYS')[1:-1].replace('\'', '').split(',')
        self.chat_api_list.append(OPENAI_KEY)
        self.chat_api_list = [api.strip() for api in self.chat_api_list if len(api) > 20]
        self.cur_api = 0
        self.file_format = args.file_format
        if args.save_image:
            self.gitee_key = self.config.get('Gitee', 'api')
        else:
            self.gitee_key = ''
        self.max_token_num = 4096
        self.encoding = tiktoken.get_encoding("gpt2")

    def get_url(self, keyword, page):
        base_url = "https://arxiv.org/search/?"
        params = {
            "query": keyword,
            "searchtype": "all",
            "abstracts": "show",
            "order": "-announced_date_first",
            "size": 50
        }
        if page > 0:
            params["start"] = page * 50
        return base_url + requests.compat.urlencode(params)

    def get_titles(self, url, days=1):
        titles = []
        links = []
        dates = []
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        articles = soup.find_all("li", class_="arxiv-result")
        today = datetime.date.today()
        last_days = datetime.timedelta(days=days)
        for article in articles:
            try:
                title = article.find("p", class_="title").text
                title = title.strip()
                link = article.find("span").find_all("a")[0].get('href')
                date_text = article.find("p", class_="is-size-7").text
                date_text = date_text.split('\n')[0].split("Submitted ")[-1].split("; ")[0]
                date_text = datetime.datetime.strptime(date_text, "%d %B, %Y").date()
                if today - date_text <= last_days:
                    titles.append(title.strip())
                    links.append(link)
                    dates.append(date_text)
            except Exception as e:
                print("error:", e)
                print("error_title:", title)
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(exc_type, fname, exc_tb.tb_lineno)
        return titles, links, dates

    def get_all_titles_from_web(self, keyword, page_num=1, days=1):
        title_list, link_list, date_list = [], [], []
        for page in range(page_num):
            url = self.get_url(keyword, page)
            titles, links, dates = self.get_titles(url, days)
            if not titles:
                break
            for title_index, title in enumerate(titles):
                print(page, title_index, title, links[title_index], dates[title_index])
            title_list.extend(titles)
            link_list.extend(links)
            date_list.extend(dates)
        print("-" * 40)
        return title_list, link_list, date_list

    def get_arxiv_web(self, args, page_num=1, days=2):
        titles, links, dates = self.get_all_titles_from_web(args.query, page_num=page_num, days=days)
        paper_list = []
        for title_index, title in enumerate(titles):
            if title_index + 1 > args.max_results:
                break
            print(title_index, title, links[title_index], dates[title_index])
            url = links[title_index] + ".pdf"
            filename = self.try_download_pdf(url, title)
            paper = Paper(path=filename, url=links[title_index], title=title)
            paper_list.append(paper)
        return paper_list

    def validateTitle(self, title):
        rstr = r"[\/\\\:\*\?\"\<\>\|]"
        new_title = re.sub(rstr, "_", title)
        return new_title

    def download_pdf(self, url, title):
        response = requests.get(url)
        date_str = str(datetime.datetime.now())[:13].replace(' ', '-')
        path = self.root_path + 'pdf_files/' + self.validateTitle(self.args.query) + '-' + date_str
        try:
            os.makedirs(path)
        except:
            pass
        filename = os.path.join(path, self.validateTitle(title)[:80] + '.pdf')
        with open(filename, "wb") as f:
            f.write(response.content)
        return filename

    @tenacity.retry(wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
                    stop=tenacity.stop_after_attempt(5),
                    reraise=True)
    def try_download_pdf(self, url, title):
        return self.download_pdf(url, title)

    def summary_with_chat(self, paper_list):
        htmls = []
        for paper_index, paper in enumerate(paper_list):
            text = ''
            text += 'Title:' + paper.title
            text += 'Url:' + paper.url
            text += 'Abstract:' + paper.abs
            text += 'Paper_info:' + paper.section_text_dict['paper_info']
            text += list(paper.section_text_dict.values())[0]
            chat_summary_text = ""
            try:
                chat_summary_text = self.chat_summary(text=text)
            except Exception as e:
                print("summary_error:", e)
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(exc_type, fname, exc_tb.tb_lineno)
                if "maximum context" in str(e):
                    current_tokens_index = str(e).find("your messages resulted in") + len(
                        "your messages resulted in") + 1
                    offset = int(str(e)[current_tokens_index:current_tokens_index + 4])
                    summary_prompt_token = offset + 1000 + 150
                    chat_summary_text = self.chat_summary(text=text, summary_prompt_token=summary_prompt_token)

            htmls.append('## Paper:' + str(paper_index + 1))
            htmls.append('\n\n\n')
            if "chat_summary_text" in locals():
                htmls.append(chat_summary_text)

            method_key = ''
            for parse_key in paper.section_text_dict.keys():
                if 'method' in parse_key.lower() or 'approach' in parse_key.lower():
                    method_key = parse_key
                    break

            chat_method_text = ""
            if method_key != '':
                text = ''
                method_text = ''
                summary_text = ''
                summary_text += "<summary>" + chat_summary_text
                method_text += paper.section_text_dict[method_key]
                text = summary_text + "\n\n<Methods>:\n\n" + method_text
                try:
                    chat_method_text = self.chat_method(text=text)
                except Exception as e:
                    print("method_error:", e)
                    exc_type, exc_obj, exc_tb = sys.exc_info()
                    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                    print(exc_type, fname, exc_tb.tb_lineno)
                    if "maximum context" in str(e):
                        current_tokens_index = str(e).find("your messages resulted in") + len(
                            "your messages resulted in") + 1
                        offset = int(str(e)[current_tokens_index:current_tokens_index + 4])
                        method_prompt_token = offset + 800 + 150
                        chat_method_text = self.chat_method(text=text, method_prompt_token=method_prompt_token)
                if "chat_method_text" in locals():
                    htmls.append(chat_method_text)
            else:
                chat_method_text = ''
            htmls.append("\n" * 4)

            conclusion_key = ''
            for parse_key in paper.section_text_dict.keys():
                if 'conclu' in parse_key.lower():
                    conclusion_key = parse_key
                    break

            text = ''
            conclusion_text = ''
            summary_text = ''
            summary_text += "<summary>" + chat_summary_text + "\n <Method summary>:\n" + chat_method_text
            chat_conclusion_text = ""
            if conclusion_key != '':
                conclusion_text += paper.section_text_dict[conclusion_key]
                text = summary_text + "\n\n<Conclusion>:\n\n" + conclusion_text
            else:
                text = summary_text
            try:
                chat_conclusion_text = self.chat_conclusion(text=text)
            except Exception as e:
                print("conclusion_error:", e)
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(exc_type, fname, exc_tb.tb_lineno)
                if "maximum context" in str(e):
                    current_tokens_index = str(e).find("your messages resulted in") + len(
                        "your messages resulted in") + 1
                    offset = int(str(e)[current_tokens_index:current_tokens_index + 4])
                    conclusion_prompt_token = offset + 800 + 150
                    chat_conclusion_text = self.chat_conclusion(text=text,
                                                                conclusion_prompt_token=conclusion_prompt_token)
            if "chat_conclusion_text" in locals():
                htmls.append(chat_conclusion_text)
            htmls.append("\n" * 4)

            date_str = str(datetime.datetime.now())[:13].replace(' ', '-')
            export_path = os.path.join(self.root_path, 'export')
            if not os.path.exists(export_path):
                os.makedirs(export_path)
            mode = 'w' if paper_index == 0 else 'a'
            file_name = os.path.join(export_path,
                                     date_str + '-' + self.validateTitle(self.query) + "." + self.file_format)
            self.export_to_markdown("\n".join(htmls), file_name=file_name, mode=mode)
            htmls = []

    @tenacity.retry(wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
                    stop=tenacity.stop_after_attempt(5),
                    reraise=True)
    def chat_conclusion(self, text, conclusion_prompt_token=800):
        openai.api_key = self.chat_api_list[self.cur_api]
        self.cur_api += 1
        self.cur_api = 0 if self.cur_api >= len(self.chat_api_list) - 1 else self.cur_api
        text_token = len(self.encoding.encode(text))
        clip_text_index = int(len(text) * (self.max_token_num - conclusion_prompt_token) / text_token)
        clip_text = text[:clip_text_index]

        messages = [
            {"role": "system",
             "content": "You are a reviewer in the field of [" + self.key_word + "] and you need to critically review this article"},
            {"role": "assistant",
             "content": "This is the <summary> and <conclusion> part of an English literature, where <summary> you have already summarized, but <conclusion> part, I need your help to summarize the following questions:" + clip_text},
            {"role": "user", "content": """
                 8. Make the following summary.Be sure to use {} answers (proper nouns need to be marked in English).
                    - (1):What is the significance of this piece of work?
                    - (2):Summarize the strengths and weaknesses of this article in three dimensions: innovation point, performance, and workload.
                    .......
                 Follow the format of the output later:
                 8. Conclusion: \n\n
                    - (1):xxx;\n
                    - (2):Innovation point: xxx; Performance: xxx; Workload: xxx;\n

                 Be sure to use {} answers (proper nouns need to be marked in English), statements as concise and academic as possible, do not repeat the content of the previous <summary>, the value of the use of the original numbers, be sure to strictly follow the format, the corresponding content output to xxx, in accordance with \n line feed, ....... means fill in according to the actual requirements, if not, you can not write.
                 """.format(self.language, self.language)},
        ]
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
        )
        result = ''
        for choice in response.choices:
            result += choice.message.content
        print("conclusion_result:\n", result)
        print("prompt_token_used:", response.usage.prompt_tokens,
              "completion_token_used:", response.usage.completion_tokens,
              "total_token_used:", response.usage.total_tokens)
        print("response_time:", response.response_ms / 1000.0, 's')
        return result

    @tenacity.retry(wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
                    stop=tenacity.stop_after_attempt(5),
                    reraise=True)
    def chat_method(self, text, method_prompt_token=800):
        openai.api_key = self.chat_api_list[self.cur_api]
        self.cur_api += 1
        self.cur_api = 0 if self.cur_api >= len(self.chat_api_list) - 1 else self.cur_api
        text_token = len(self.encoding.encode(text))
        clip_text_index = int(len(text) * (self.max_token_num - method_prompt_token) / text_token)
        clip_text = text[:clip_text_index]
        messages = [
            {"role": "system",
             "content": "You are a researcher in the field of [" + self.key_word + "] who is good at summarizing papers using concise statements"},
            {"role": "assistant",
             "content": "This is the <summary> and <Method> part of an English document, where <summary> you have summarized, but the <Methods> part, I need your help to read and summarize the following questions." + clip_text},
            {"role": "user", "content": """
                 7. Describe in detail the methodological idea of this article. Be sure to use {} answers (proper nouns need to be marked in English). For example, its steps are.
                    - (1):...
                    - (2):...
                    - (3):...
                    - .......
                 Follow the format of the output that follows:
                 7. Methods: \n\n
                    - (1):xxx;\n
                    - (2):xxx;\n
                    - (3):xxx;\n
                    ....... \n\n

                 Be sure to use {} answers (proper nouns need to be marked in English), statements as concise and academic as possible, do not repeat the content of the previous <summary>, the value of the use of the original numbers, be sure to strictly follow the format, the corresponding content output to xxx, in accordance with \n line feed, ....... means fill in according to the actual requirements, if not, you can not write.
                 """.format(self.language, self.language)},
        ]
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
        )
        result = ''
        for choice in response.choices:
            result += choice.message.content
        print("method_result:\n", result)
        print("prompt_token_used:", response.usage.prompt_tokens,
              "completion_token_used:", response.usage.completion_tokens,
              "total_token_used:", response.usage.total_tokens)
        print("response_time:", response.response_ms / 1000.0, 's')
        return result

    @tenacity.retry(wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
                    stop=tenacity.stop_after_attempt(5),
                    reraise=True)
    def chat_summary(self, text, summary_prompt_token=1100):
        openai.api_key = self.chat_api_list[self.cur_api]
        self.cur_api += 1
        self.cur_api = 0 if self.cur_api >= len(self.chat_api_list) - 1 else self.cur_api
        text_token = len(self.encoding.encode(text))
        clip_text_index = int(len(text) * (self.max_token_num - summary_prompt_token) / text_token)
        clip_text = text[:clip_text_index]
        messages = [
            {"role": "system",
             "content": "You are a researcher in the field of [" + self.key_word + "] who is good at summarizing papers using concise statements"},
            {"role": "assistant",
             "content": "This is the title, author, link, abstract and introduction of an English document. I need your help to read and summarize the following questions: " + clip_text},
            {"role": "user", "content": """
                 1. Mark the title of the paper (with Chinese translation)
                 2. list all the authors and their institution affiliations.
                 3. Summarize the paper in a concise manner, listing the following information.
                    - (1):What is the motivation of this research?
                    - (2):What are the problems with the past methods? What are the problems with them? Is the approach well motivated?
                    - (3):What is the research methodology proposed in this paper?
                    - (4):On what task and what performance is achieved by the methods in this paper? Can the performance support their goals?
                 Follow the format of the output that follows:
                 1. Title: xxx\n\n
                 2. Authors: xxx\n\n
                 3. Affiliation: xxx\n\n
                 4. Keywords: xxx\n\n
                 5. Urls: xxx or xxx , xxx \n\n
                 6. Summary: \n\n
                    - (1):xxx;\n
                    - (2):xxx;\n
                    - (3):xxx;\n
                    - (4):xxx.\n\n

                 Be sure to use {} answers (proper nouns need to be marked in English), statements as concise and academic as possible, do not have too much repetitive information, numerical values using the original numbers, be sure to strictly follow the format, the corresponding content output to xxx, in accordance with \n line feed.
                 """.format(self.language, self.language)},
        ]
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
        )
        result = ''
        for choice in response.choices:
            result += choice.message.content
        print("summary_result:\n", result)
        print("prompt_token_used:", response.usage.prompt_tokens,
              "completion_token_used:", response.usage.completion_tokens,
              "total_token_used:", response.usage.total_tokens)
        print("response_time:", response.response_ms / 1000.0, 's')
        return result

    def export_to_markdown(self, text, file_name, mode='w'):
        with open(file_name, mode, encoding="utf-8") as f:
            f.write(text)

    def show_info(self):
        print(f"Key word: {self.key_word}")
        print(f"Query: {self.query}")
        print(f"Sort: {self.sort}")

def chat_arxiv_main(args):
    reader1 = Reader(key_word=args.key_word, query=args.query, args=args)
    reader1.show_info()
    paper_list = reader1.get_arxiv_web(args=args, page_num=args.page_num, days=args.days)
    reader1.summary_with_chat(paper_list=paper_list)

# Funções auxiliares
def load_agent_options() -> list:
    agent_options = ['Escolher um especialista...']
    if os.path.exists(FILEPATH):
        with open(FILEPATH, 'r') as file:
            try:
                agents = json.load(file)
                agent_options.extend([agent["agente"] for agent in agents if "agente" in agent])
            except json.JSONDecodeError:
                st.error("Erro ao ler o arquivo de Agentes 4  -. Por favor, verifique o formato.")
    return agent_options

def get_max_tokens(model_name: str) -> int:
    return MODEL_MAX_TOKENS.get(model_name, 4096)

def refresh_page():
    st.rerun()

def save_expert(expert_title: str, expert_description: str):
    with open(FILEPATH, 'r+') as file:
        agents = json.load(file) if os.path.getsize(FILEPATH) > 0 else []
        agents.append({"agente": expert_title, "descricao": expert_description})
        file.seek(0)
        json.dump(agents, file, indent=4)
        file.truncate()

def fetch_assistant_response(user_input: str, user_prompt: str, model_name: str, temperature: float, agent_selection: str, groq_api_key: str) -> Tuple[str, str]:
    phase_two_response = ""
    expert_title = ""
    try:
        client = Groq(api_key=groq_api_key)
        def get_completion(prompt: str) -> str:
            completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "Você é um assistente útil."},
                    {"role": "user", "content": prompt},
                ],
                model=model_name,
                temperature=temperature,
                max_tokens=get_max_tokens(model_name),
                top_p=1,
                stop=None,
                stream=False
            )
            return completion.choices[0].message.content
        if agent_selection == "Escolha um especialista...":
            phase_one_prompt = (
                "Você é um assistente de pesquisa de alta precisão e profundidade."
                "Determine o especialista mais adequado para responder à solicitação: {user_input} e {user_prompt}."
                "Forneça um título e uma descrição detalhada das habilidades do especialista."
            )
            phase_one_response = get_completion(phase_one_prompt)
            first_period_index = phase_one_response.find(".")
            expert_title = phase_one_response[:first_period_index].strip()
            expert_description = phase_one_response[first_period_index + 1:].strip()
            save_expert(expert_title, expert_description)
        else:
            with open(FILEPATH, 'r') as file:
                agents = json.load(file)
                agent_found = next((agent for agent in agents if agent["agente"] == agent_selection), None)
                if agent_found:
                    expert_title = agent_found["agente"]
                    expert_description = agent_found["descricao"]
                else:
                    raise ValueError("Especialista selecionado não encontrado no arquivo.")
        phase_two_prompt = (
            f"Você é {expert_title}, um especialista renomado. Forneça uma resposta detalhada e abrangente para a solicitação: {user_input} e {user_prompt}."
            f"Use sua experiência para abordar todos os aspectos relevantes da questão."
        )
        phase_two_response = get_completion(phase_two_prompt)
    except Exception as e:
        st.error(f"Ocorreu um erro: {e}")
        return "", ""
    return expert_title, phase_two_response

def refine_response(expert_title: str, phase_two_response: str, user_input: str, user_prompt: str, model_name: str, temperature: float, groq_api_key: str, references_file: str) -> str:
    try:
        client = Groq(api_key=groq_api_key)
        def get_completion(prompt: str) -> str:
            completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "Você é um assistente útil."},
                    {"role": "user", "content": prompt},
                ],
                model=model_name,
                temperature=temperature,
                max_tokens=get_max_tokens(model_name),
                top_p=1,
                stop=None,
                stream=False
            )
            return completion.choices[0].message.content
        refine_prompt = (
            f"Refine a seguinte resposta fornecida por {expert_title} com base na análise e melhoria do conteúdo: {phase_two_response}"
            f"Inclua todas as informações relevantes e garanta a precisão: {user_input} e {user_prompt}."
        )
        if not references_file:
            refine_prompt += (
                f"\n\nDevido à ausência de referências fornecidas, certifique-se de fornecer uma resposta detalhada e precisa, "
                f"mesmo sem o uso de fontes externas."
            )
        refined_response = get_completion(refine_prompt)
        return refined_response
    except Exception as e:
        st.error(f"Ocorreu um erro durante o refinamento: {e}")
        return ""

def evaluate_response_with_rag(user_input: str, user_prompt: str, expert_description: str, assistant_response: str, model_name: str, temperature: float, groq_api_key: str) -> str:
    try:
        client = Groq(api_key=groq_api_key)
        def get_completion(prompt: str) -> str:
            completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "Você é um assistente útil."},
                    {"role": "user", "content": prompt},
                ],
                model=model_name,
                temperature=temperature,
                max_tokens=get_max_tokens(model_name),
                top_p=1,
                stop=None,
                stream=False
            )
            return completion.choices[0].message.content
        rag_prompt = (
            f"Avalie a seguinte resposta usando o Rational Agent Generator (RAG): {assistant_response}"
            f"Baseie a avaliação na descrição do especialista: {expert_description}."
            f"Forneça uma análise detalhada e abrangente considerando a solicitação: {user_input} e {user_prompt}."
        )
        rag_response = get_completion(rag_prompt)
        return rag_response
    except Exception as e:
        st.error(f"Ocorreu um erro durante a avaliação com RAG: {e}")
        return ""

# Carregar as opções de especialistas do arquivo JSON
agent_options = load_agent_options()

# Interface do Streamlit
st.title("Consulta ao arXiv com Resumo e Avaliação de Especialistas")

user_input = st.text_area("Digite sua solicitação:")
user_prompt = st.text_area("Digite o prompt adicional (opcional):")
agent_selection = st.selectbox("Escolha um Especialista", options=agent_options)
model_name = st.selectbox("Escolha um Modelo", list(MODEL_MAX_TOKENS.keys()))
temperature = st.slider("Nível de Criatividade", 0.0, 1.0, 0.5)
groq_api_key = st.text_input("Chave da API Groq")

if st.button("Buscar Resposta"):
    expert_title, response = fetch_assistant_response(user_input, user_prompt, model_name, temperature, agent_selection, groq_api_key)
    st.session_state.expert_title = expert_title
    st.session_state.response = response
    st.write(f"Especialista: {expert_title}")
    st.write(f"Resposta: {response}")

if st.button("Refinar Resposta"):
    if 'response' in st.session_state:
        refined_response = refine_response(st.session_state.expert_title, st.session_state.response, user_input, user_prompt, model_name, temperature, groq_api_key, None)
        st.session_state.refined_response = refined_response
        st.write(f"Resposta Refinada: {refined_response}")
    else:
        st.warning("Por favor, busque uma resposta antes de refinar.")

if st.button("Avaliar com RAG"):
    if 'response' in st.session_state:
        rag_response = evaluate_response_with_rag(user_input, user_prompt, st.session_state.expert_title, st.session_state.response, model_name, temperature, groq_api_key)
        st.session_state.rag_response = rag_response
        st.write(f"Avaliação com RAG: {rag_response}")
    else:
        st.warning("Por favor, busque uma resposta antes de avaliar com RAG.")
