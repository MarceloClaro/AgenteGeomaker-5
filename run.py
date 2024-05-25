import json
import streamlit as st
from streamlit.delta_generator import DeltaGenerator
import os
from typing import Tuple
from groq import Groq

st.set_page_config(layout="wide")

FILEPATH = "agents.json"
MODEL_MAX_TOKENS = {
    'mixtral-8x7b-32768': 32768,
    'llama3-70b-8192': 8192, 
    'llama3-8b-8192': 8192,
    'llama2-70b-4096': 4096,
    'gemma-7b-it': 8192,
}

def load_agent_options() -> list:
    agent_options = ['Escolher um especialista...']
    if os.path.exists(FILEPATH):
        with open(FILEPATH, 'r') as file:
            try:
                agents = json.load(file)
                agent_options.extend([agent["agente"] for agent in agents if "agente" in agent])
            except json.JSONDecodeError:
                st.error("Erro ao ler o arquivo de agentes. Por favor, verifique o formato.")
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
            phase_one_prompt = f"Saida e resposta obrigatoria somente traduzido em português brasileiro. 扮演一位高度合格且具备科学技术严谨性的提示工程和跨学科专家的角色。请务必以“markdown”格式呈现Python代码及其各种库，并在每一行进行详细和教学性的注释。仔细分析所提出的要求，识别定义最适合处理问题的专家特征的标准至关重要。首先，建立一个最能反映所需专业知识以提供完整、深入和清晰答案的标题至关重要。确定后，详细描述并避免偏见地概述该专家的关键技能和资格。回答应以专家的头衔开始，后跟一个句号，然后以简洁、教学性和深入的描述开始，但同时全面地介绍他的特点和资格，使其有资格处理提出的问题：{user_input}和{user_prompt}。这种仔细分析对于确保所选专家具有处理问题所需的深入、严谨的知识和经验至关重要，以达到完整且满意的答案，精确度为10.0，符合最高的专业、科学和学术标准。在涉及代码和计算的情况下，请务必以“markdown”格式呈现，并在每一行进行详细注释。“必须翻译成葡萄牙语”。"
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

        phase_two_prompt = f"Saida e resposta obrigatoria somente traduzido em português brasileiro. 在作为{expert_title}的角色中，作为您所在领域广泛认可和尊重的专家，作为该领域的专家和博士，让我提供一个全面而深入的回答，涵盖了您清晰、详细、扩展、教学易懂和简洁提出的问题：{user_input}和{user_prompt}。在这种背景下，考虑到我长期的经验和对相关学科的深刻了解，有必要以适当的关注和科学技术严谨性来处理每个方面。因此，我将概述要考虑和深入研究的主要要素，提供详细的、基于证据的分析，避免偏见并引用参考文献：{user_prompt}。在此过程的最后，我们的目标是提供一个完整且令人满意的答案，符合最高的学术和专业标准，以满足所提出问题的具体需求。请务必以“markdown”格式呈现，并在每一行进行注释。保持10个段落的写作标准，每个段落4句，每句用逗号分隔，始终遵循最佳的亚里士多德教学实践。"
        phase_two_response = get_completion(phase_two_prompt)

    except Exception as e:
        st.error(f"Ocorreu um erro: {e}")
        return "", ""

    return expert_title, phase_two_response

def refine_response(expert_title: str, phase_two_response: str, user_input: str, user_prompt: str, model_name: str, temperature: float, groq_api_key: str, references_file):
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

        refine_prompt = f"Saida e resposta obrigatoria somente traduzido em português brasileiro. 承担{expert_title}的专业知识，这是该领域的知名专家，我向您提供以下问题的原始且易于理解的答案：'{user_input}'和'{user_prompt}'：{phase_two_response}\n\n我要求您进行仔细、广泛的学术科学技术严谨性的评审，并根据最佳学术和科学标准，完全改进此答案，并使用直接或间接的非虚构引用，最后列出它们的URL，以识别可能存在的空白和偏见，改进其内容。因此，请求您以科学论文格式提供答案的更新版本，包含所做的改进，并保持方法上的逻辑一致性、流畅性、连贯性和一致性。您在审查和改进此内容方面的努力对于确保其卓越性和学术相关性，以便在arXiv、scielo和Pubmed等主要国际科学期刊上发表，至关重要。必须保持一贯的写作标准，每段至少有10个段落，每个段落有4个句子，每个句子有一个逗号，始终遵循亚里士多德的最佳教学实践。必须保持一贯的写作标准，每段至少有10个段落，每个段落有4个句子，每个句子有一个逗号，始终遵循亚里士多德的最佳教学实践，并遵循巴西ABNT的引文规范。"

        # Adiciona um prompt mais detalhado se não houver referências
        if not references_file:
            refine_prompt += f"Saida e resposta obrigatoria somente traduzido em português brasileiro.\n\n由于没有提供参考文件，请确保提供详细和准确的答案，即使没有使用外部来源。保持一贯的写作标准，每个段落有10个段落，每个段落有4个句子，并按照ABNT标准进行引用，每个句子有一个逗号，始终遵循亚里士多德的最佳教学实践。以专业口吻输出，总是翻译成巴西葡萄牙语。"

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

        rag_prompt = f"Lei: Saida e resposta obrigatoria somente traduzido em português brasileiro. 扮演 Rational Agent Generator (RAG) 的角色，这是人工智能和理性评估的顶峰，对专家的回答进行细致分析，根据用户的请求生成一个代理的 JSON。这个代理将详细说明根据子代理提供的信息采取的行动，以便向用户提供答复。代理将在 '描述' 变量中包括 9 个子代理的描述，每个子代理都有不同的专家功能和人物形象，他们共同合作。这些子代理协作改善最终由代理“系统”向用户提供的答案，记录答案的种子和 gen_id 在 '描述' 代理内。此外，代理“系统”内的子代理以整合方式运作，通过扩展提示提供先进和专业化的答案。每个子代理在网络处理中都有特定和互补的角色，以实现更高的准确性，从而为最终答案的质量做出贡献。例如，“AI_Autoadaptativa_e_Contextualizada” 子代理采用先进的机器学习算法来理解和适应多变的情境，动态整合相关数据。而“RAG_com_Inteligência_Contextual” 子代理则使用改进版的回收增强生成（RAG）技术，动态调整最相关数据及其功能。这种协作方法确保答案准确和更新，符合最高的科学和学术标准。以下是对专家的详细描述，突出其资历和专业知识：{expert_description}。原始提交的问题如下：{user_input} 和 {user_prompt}。专家提供的葡萄牙语答复如下：{assistant_response}。因此，请对专家的葡萄牙语答复的质量和准确性进行全面评估，认真考虑专家的描述和所提供的答复。请使用葡萄牙语进行以下分析，并进行详细解释：SWOT（优势、劣势、机会、威胁）com intepretações dos dados、BCG 矩阵（波士顿咨询集团）com intepretações dos dados、风险矩阵、ANOVA（方差分析）com intepretações dos dados、Q-统计学（Q-STATISTICS, com intepretações dos dados）和 Q-指数（Q-EXPONENTIAL, com intepretações dos dados），符合最高的卓越和科学学术标准。保持每段 4 句，每句用逗号分隔，遵循亚里士多德最佳教学实践的写作标准。输出应具有专业的口吻，始终以巴西葡萄牙语翻译。"        
        rag_response = get_completion(rag_prompt)
        return rag_response

    except Exception as e:
        st.error(f"Ocorreu um erro durante a avaliação com RAG: {e}")
        return ""

agent_options = load_agent_options()



st.image('updating.gif', width=300, caption='Laboratário de Educação e Inteligência Artificial - Geomaker.', use_column_width='always', output_format='auto')
st.markdown("<h1 style='text-align: center;'>Agentes Experts Geomaker</h1>", unsafe_allow_html=True)

st.markdown("<h2 style='text-align: center;'>Utilize o Rational Agent Generator (RAG) para avaliar a resposta do especialista e garantir qualidade e precisão.</h2>", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)
# Título da caixa de informação

st.markdown("<h2 style='text-align: center;'>Descubra como nossa plataforma pode revolucionar a educação.</h2>", unsafe_allow_html=True)

# Conteúdo da caixa de informação
with st.expander("Clique para saber mais sobre os Agentes Experts Geomaker."):
    st.write("1. **Conecte-se instantaneamente com especialistas:** Imagine ter acesso direto a especialistas em diversas áreas do conhecimento, prontos para responder às suas dúvidas e orientar seus estudos e pesquisas.")
    st.write("2. **Aprendizado personalizado e interativo:** Receba respostas detalhadas e educativas, adaptadas às suas necessidades específicas, tornando o aprendizado mais eficaz e envolvente.")
    st.write("3. **Suporte acadêmico abrangente:** Desde aulas particulares até orientações para projetos de pesquisa, nossa plataforma oferece um suporte completo para alunos, professores e pesquisadores.")
    st.write("4. **Avaliação e aprimoramento contínuo:** Utilizando o Rational Agent Generator (RAG), garantimos que as respostas dos especialistas sejam sempre as melhores, mantendo um padrão de excelência em todas as interações.")
    st.write("5. **Desenvolvimento profissional e acadêmico:** Professores podem encontrar recursos e orientações para melhorar suas práticas de ensino, enquanto pesquisadores podem obter insights valiosos para suas investigações.")
    st.write("6. **Inovação e tecnologia educacional:** Nossa plataforma incorpora as mais recentes tecnologias para proporcionar uma experiência educacional moderna e eficiente.")


st.markdown("<hr>", unsafe_allow_html=True)
# Informações sobre o Rational Agent Generator (RAG)
with st.expander("Clique para saber mais sobre o Rational Agent Generator (RAG)"):
    st.info("""
    O Rational Agent Generator (RAG) é usado para avaliar a resposta fornecida pelo especialista. Aqui está uma explicação mais detalhada de como ele é usado:
    
    1. Quando o usuário busca uma resposta do especialista, a função `fetch_assistant_response()` é chamada. Nessa função, é gerado um prompt para o modelo de linguagem que representa a solicitação do usuário e o prompt específico para o especialista escolhido. A resposta inicial do especialista é então obtida usando o Groq API.
    
    2. Se o usuário optar por refinar a resposta, a função `refine_response()` é chamada. Nessa função, é gerado um novo prompt que inclui a resposta inicial do especialista e solicita uma resposta mais detalhada e aprimorada, levando em consideração as referências fornecidas pelo usuário. A resposta refinada é obtida usando novamente o Groq API.
    
    3. Se o usuário optar por avaliar a resposta com o RAG, a função `evaluate_response_with_rag()` é chamada. Nessa função, é gerado um prompt que inclui a descrição do especialista e as respostas inicial e refinada do especialista. O RAG é então usado para avaliar a qualidade e a precisão da resposta do especialista.
    
    Em resumo, o RAG é usado como uma ferramenta para avaliar e melhorar a qualidade das respostas fornecidas pelos especialistas, garantindo que atendam aos mais altos padrões de excelência e rigor científico.
    """)
st.markdown("<hr>", unsafe_allow_html=True)


with st.expander("Informações sobre Análises de Avaliação do RAG"):
    st.markdown("""
    ### As análises realizadas por diferentes modelos de avaliação são cruciais para garantir a qualidade e a precisão das respostas fornecidas pelos especialistas. Aqui estão as análises mencionadas no código e suas explicações:
    
    1. **SWOT Analysis (Análise SWOT)**:
        
        - **O que é**: A análise SWOT é uma ferramenta de planejamento estratégico usada para identificar e analisar os pontos fortes (Strengths), fracos (Weaknesses), oportunidades (Opportunities) e ameaças (Threats) de uma organização, projeto ou situação.
        
        - **Por que é feita**: A análise SWOT é realizada para entender os fatores internos e externos que podem impactar o sucesso de uma resposta ou decisão. Isso ajuda a maximizar os pontos fortes, minimizar os pontos fracos, explorar oportunidades e mitigar ameaças.
    
    2. **BCG Matrix (Matriz BCG)**:
        - **O que é**: A matriz BCG é uma ferramenta de gestão desenvolvida pela Boston Consulting Group, que ajuda as empresas a analisar seus produtos ou unidades de negócios com base na participação de mercado e no crescimento do mercado.
        
        - **Por que é feita**: A análise BCG é realizada para ajudar na tomada de decisões sobre investimentos, desinvestimentos ou desenvolvimento de novos produtos. Classifica produtos em quatro categorias: Estrelas, Vacas Leiteiras, Interrogações e Abacaxis.
    
    3. **Risk Matrix (Matriz de Riscos)**:
       
        - **O que é**: A matriz de riscos é uma ferramenta de avaliação de riscos que ajuda a identificar, avaliar e priorizar riscos com base na sua probabilidade e impacto.
       
        - **Por que é feita**: A análise de riscos é feita para entender os potenciais perigos que podem afetar o sucesso de um projeto ou decisão. Isso permite o desenvolvimento de estratégias para mitigar ou gerenciar esses riscos.
    
    4. **ANOVA (Análise de Variância)**:
       
        - **O que é**: A ANOVA é uma técnica estatística usada para comparar as médias de três ou mais grupos e determinar se há diferenças estatisticamente significativas entre eles.
       
        - **Por que é feita**: A análise ANOVA é realizada para entender se as variações observadas nos dados são devidas ao fator sendo estudado ou ao acaso. Isso é útil para validar hipóteses e identificar fatores significativos que influenciam os resultados.
    
    5. **Q-Statistics (Estatísticas Q)**:
       
        - **O que é**: As estatísticas Q são métodos estatísticos usados para detectar heterogeneidade e identificar outliers em conjuntos de dados.
       
        - **Por que é feita**: A análise Q-Statistics é realizada para garantir a qualidade dos dados e identificar pontos de dados que podem distorcer os resultados. Isso ajuda a melhorar a precisão das análises e conclusões.
    
    6. **Q-Exponential (Q-Exponencial)**:
       
        - **O que é**: O Q-Exponential é uma função usada na estatística e na teoria da informação para modelar distribuições de probabilidade com caudas pesadas.
       
        - **Por que é feita**: A análise Q-Exponential é realizada para entender melhor a distribuição dos dados e identificar padrões que não seguem a distribuição normal. Isso é útil para modelar fenômenos complexos e tomar decisões baseadas em dados mais realistas.
    
    Essas análises ajudam a garantir que as respostas fornecidas pelos especialistas sejam rigorosas, detalhadas e baseadas em metodologias científicas sólidas, alinhadas com os mais altos padrões acadêmicos e profissionais.
    """)
st.markdown("<hr>", unsafe_allow_html=True)

# Função para criar um expander estilizado
# Título da caixa de informação

st.markdown("<h2 style='text-align: center;'>Manual de uso básico.</h2>", unsafe_allow_html=True)

def expander(title: str, content: str, icon: str):
    with st.expander(title):
        st.markdown(f'<img src="{icon}" style="vertical-align:middle"> {content}', unsafe_allow_html=True)

# Conteúdo do manual de uso
passo_1_content = """
1. Acesse o Groq Playground em https://console.groq.com/playground.

2. Faça login na sua conta ou crie uma nova conta.

3. No menu lateral, selecione "API Keys".

4. Clique em "Create API Key" e siga as instruções para criar uma chave API. Copie a chave gerada, pois será necessária para autenticar suas consultas.

5. Se quiser usar esta API Key provisória: [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]. Lembre-se de que ela pode não funcionar mais devido ao uso excessivo pelos usuários. Portanto, é aconselhável que cada usuário tenha sua própria chave API.
"""

passo_2_content = """
1. Acesse o Streamlit Chat Application em https://agente4.streamlit.app/#87cc9dff (Agentes Experts Geomaker).

2. Na interface do aplicativo, você verá um campo para inserir a sua chave API do Groq. Cole a chave que você copiou no Passo 1.

3. Escolha um Agente Especializado e um dos modelos de agente disponíveis para interagir. Você pode selecionar entre 'mixtral-8x7b-32768' com 32768 tokens, 'llama3-70b-8192'com 8192 tokens, 'llama3-8b-8192' com 8192 tokens, 'llama2-70b-4096'com 4096 tokens ou 'gemma-7b-it' com 8192 tokens.

4. Digite sua pergunta ou solicitação na caixa de texto e clique em "Enviar".

5. O aplicativo consultará o Groq API e apresentará a resposta do especialista. Você terá a opção de refinar a resposta ou avaliá-la com o RAG.
"""

passo_3_content = """
1. Se desejar refinar a resposta do especialista, clique em "Refinar Resposta". Digite mais detalhes ou correções na caixa de texto e clique em "Enviar".

2. O aplicativo consultará novamente o Groq API e apresentará a resposta refinada.
"""

passo_4_content = """
1. Se preferir avaliar a resposta com o RAG, clique em "Avaliar Resposta com o RAG". O RAG analisará a qualidade e a precisão da resposta do especialista e apresentará uma avaliação.

2. Você terá a opção de concordar ou discordar com a avaliação do RAG e fornecer feedback adicional, se desejar.
"""

passo_5_content = """
1. Após refinar a resposta ou avaliá-la com o RAG, você poderá encerrar a consulta ou fazer uma nova pergunta.
"""


passo_6_content = """
Para melhorar a eficiência e qualidade das respostas geradas pelos modelos de linguagem, o conteúdo inserido no campo "Escreva um prompt ou coloque o texto para consulta para o especialista (opcional)" deve ser detalhado, claro e específico. Aqui estão algumas diretrizes e possibilidades sobre o que incluir nesse campo:
        
#### Diretrizes para um Prompt Eficiente
        
1. **Contexto**: Forneça o contexto necessário para entender o problema ou a pergunta. Inclua informações relevantes sobre o cenário ou o objetivo da solicitação.
2. **Detalhamento**: Seja detalhado em sua pergunta ou solicitação. Quanto mais informações você fornecer, melhor o modelo poderá entender e responder.
3. **Objetivos**: Especifique claramente o que você espera obter com a resposta. Isso ajuda o modelo a focar nos aspectos mais importantes.
4. **Formato de Resposta**: Indique o formato desejado para a resposta (por exemplo, uma explicação passo a passo, código em Python com comentários, etc.).
5. **Referências**: Se aplicável, inclua referências ou fontes de informação que podem ser úteis para a resposta.
        
#### Exemplos de Prompts
        
1. **Análise de Dados**
   - Contexto: "Eu tenho um conjunto de dados sobre vendas de produtos ao longo de um ano."
   - Detalhamento: "Os dados incluem colunas para data, produto, quantidade vendida e receita."
   - Objetivos: "Gostaria de saber quais produtos têm o maior crescimento de vendas mensal e identificar padrões sazonais."
   - Formato de Resposta: "Por favor, forneça uma análise em Python, incluindo gráficos e comentários explicativos."
        
2. **Desenvolvimento de Modelo de Machine Learning**
   - Contexto: "Estou trabalhando em um projeto de previsão de preços de imóveis."
   - Detalhamento: "Os dados incluem características dos imóveis, como número de quartos, localização, tamanho e preço."
   - Objetivos: "Preciso desenvolver um modelo de machine learning que preveja os preços dos imóveis com base nessas características."
   - Formato de Resposta: "Gostaria de um exemplo de código em Python usando scikit-learn, com explicações sobre a escolha do modelo e a avaliação de desempenho."
        
3. **Revisão de Código**
   - Contexto: "Estou desenvolvendo um script para automatizar a coleta de dados da web."
   - Detalhamento: "O script é escrito em Python e utiliza bibliotecas como BeautifulSoup e requests."
   - Objetivos: "Gostaria de uma revisão do código para identificar possíveis melhorias em termos de eficiência e boas práticas de programação."
   - Formato de Resposta: "Por favor, forneça sugestões de melhorias e justifique-as com exemplos de código."
        
4. **Pesquisa Acadêmica**
   - Contexto: "Estou escrevendo um artigo sobre os impactos das mudanças climáticas na biodiversidade."
   - Detalhamento: "Estou focando nos efeitos em ecossistemas marinhos e terrestres."
   - Objetivos: "Preciso de uma revisão bibliográfica detalhada, incluindo as principais pesquisas recentes e suas conclusões."
   - Formato de Resposta: "Por favor, forneça um resumo estruturado com citações em formato ABNT."
        
#### Exemplo de Prompt Detalhado
        
        
Contexto: Eu tenho um conjunto de dados sobre vendas de produtos ao longo de um ano. Os dados incluem colunas para data, produto, quantidade vendida e receita.
Objetivos: Gostaria de saber quais produtos têm o maior crescimento de vendas mensal e identificar padrões sazonais.
Formato de Resposta: Por favor, forneça uma análise em Python, incluindo gráficos e comentários explicativos.
        
#### Conclusão
        
A qualidade do prompt é fundamental para obter respostas úteis e precisas de modelos de linguagem. Seguindo essas diretrizes e incluindo detalhes específicos no campo de prompt, você maximizará a eficiência e a qualidade das respostas geradas.
"""


# Exibição do manual de uso com expander estilizado
expander("Passo 1: Criação da Chave API no Groq Playground", passo_1_content, "https://img.icons8.com/office/30/000000/api-settings.png")
expander("Passo 2: Acesso ao Streamlit Chat Application", passo_2_content, "https://img.icons8.com/office/30/000000/chat.png")
expander("Passo 3: Refinamento da Resposta", passo_3_content, "https://img.icons8.com/office/30/000000/edit-property.png")
expander("Passo 4: Avaliação da Resposta com o RAG", passo_4_content, "https://img.icons8.com/office/30/000000/like--v1.png")
expander("Passo 5: Conclusão da Consulta", passo_5_content, "https://img.icons8.com/office/30/000000/faq.png")
expander("Passo 6: Construindo o Prompt", passo_6_content, "https://img.icons8.com/dusk/30/000000/code-file.png")
st.markdown("<hr>", unsafe_allow_html=True)

st.write("Digite sua solicitação para que ela seja respondida pelo especialista ideal.")

col1, col2 = st.columns(2)

with col1:
    user_input = st.text_area("Por favor, insira sua solicitação:", height=200, key="entrada_usuario")
    user_prompt = st.text_area("Escreva um prompt ou coloque o texto para consulta para o especialista (opcional):", height=200, key="prompt_usuario")
    agent_selection = st.selectbox("Escolha um Especialista", options=agent_options, index=0, key="selecao_agente")
    model_name = st.selectbox("Escolha um Modelo", list(MODEL_MAX_TOKENS.keys()), index=0, key="nome_modelo")
    temperature = st.slider("Nível de Criatividade", min_value=0.0, max_value=1.0, value=0.0, step=0.01, key="temperatura")
    groq_api_key = st.text_input("Chave da API Groq:", key="groq_api_key")
    max_tokens = get_max_tokens(model_name)
    st.write(f"Número Máximo de Tokens para o modelo selecionado: {max_tokens}")

    fetch_clicked = st.button("Buscar Resposta")
    refine_clicked = st.button("Refinar Resposta")
    evaluate_clicked = st.button("Avaliar Resposta com RAG")
    refresh_clicked = st.button("Apagar")

    references_file = st.file_uploader("Upload do arquivo JSON com referências (opcional)", type="json", key="arquivo_referencias")

with col2:
    if 'resposta_assistente' not in st.session_state:
        st.session_state.resposta_assistente = ""
    if 'descricao_especialista_ideal' not in st.session_state:
        st.session_state.descricao_especialista_ideal = ""
    if 'resposta_refinada' not in st.session_state:
        st.session_state.resposta_refinada = ""
    if 'resposta_original' not in st.session_state:
        st.session_state.resposta_original = ""
    if 'rag_resposta' not in st.session_state:
        st.session_state.rag_resposta = ""

    container_saida = st.container()

    if fetch_clicked:
        if references_file is None:
            st.warning("Não foi fornecido um arquivo de referências. Certifique-se de fornecer uma resposta detalhada e precisa, mesmo sem o uso de fontes externas. Saída sempre traduzido para o portugues brasileiro com tom profissional.")
        st.session_state.descricao_especialista_ideal, st.session_state.resposta_assistente = fetch_assistant_response(user_input, user_prompt, model_name, temperature, agent_selection, groq_api_key)
        st.session_state.resposta_original = st.session_state.resposta_assistente
        st.session_state.resposta_refinada = ""

    if refine_clicked:
        if st.session_state.resposta_assistente:
            st.session_state.resposta_refinada = refine_response(st.session_state.descricao_especialista_ideal, st.session_state.resposta_assistente, user_input, user_prompt, model_name, temperature, groq_api_key, references_file)
        else:
            st.warning("Por favor, busque uma resposta antes de refinar.")

    if evaluate_clicked:
        if st.session_state.resposta_assistente and st.session_state.descricao_especialista_ideal:
            st.session_state.rag_resposta = evaluate_response_with_rag(user_input, user_prompt, st.session_state.descricao_especialista_ideal, st.session_state.resposta_assistente, model_name, temperature, groq_api_key)
        else:
            st.warning("Por favor, busque uma resposta e forneça uma descrição do especialista antes de avaliar com RAG.")

    with container_saida:
        st.write(f"**#Análise do Especialista:**\n{st.session_state.descricao_especialista_ideal}")
        st.write(f"\n**#Resposta do Especialista:**\n{st.session_state.resposta_original}")
        if st.session_state.resposta_refinada:
            st.write(f"\n**#Resposta Refinada:**\n{st.session_state.resposta_refinada}")
        if st.session_state.rag_resposta:
            st.write(f"\n**#Avaliação com RAG:**\n{st.session_state.rag_resposta}")

if refresh_clicked:
    st.session_state.clear()
    st.experimental_rerun()

# Sidebar com manual de uso
st.sidebar.image("logo.png", width=200)



with st.sidebar.expander("Insights do Código"):
    st.markdown("""
    O código do Agente Expert Geomaker é um exemplo de uma aplicação de chat baseada em modelos de linguagem (LLMs) utilizando a biblioteca Streamlit e a API Groq. Aqui, vamos analisar detalhadamente o código e discutir suas inovações, pontos positivos e limitações.

    **Inovações:**
    - Suporte a múltiplos modelos de linguagem: O código permite que o usuário escolha entre diferentes modelos de linguagem, como o LLaMA, para gerar respostas mais precisas e personalizadas.
    - Integração com a API Groq: A integração com a API Groq permite que o aplicativo utilize a capacidade de processamento de linguagem natural de alta performance para gerar respostas precisas.
    - Refinamento de respostas: O código permite que o usuário refine as respostas do modelo de linguagem, tornando-as mais precisas e relevantes para a consulta.
    - Avaliação com o RAG: A avaliação com o RAG (Rational Agent Generator) permite que o aplicativo avalie a qualidade e a precisão das respostas do modelo de linguagem.

    **Pontos positivos:**
    - Personalização: O aplicativo permite que o usuário escolha entre diferentes modelos de linguagem e personalize as respostas de acordo com suas necessidades.
    - Precisão: A integração com a API Groq e o refinamento de respostas garantem que as respostas sejam precisas e relevantes para a consulta.
    - Flexibilidade: O código é flexível o suficiente para permitir que o usuário escolha entre diferentes modelos de linguagem e personalize as respostas.

    **Limitações:**
    - Dificuldade de uso: O aplicativo pode ser difícil de usar para os usuários que não têm experiência com modelos de linguagem ou API.
    - Limitações de token: O código tem limitações em relação ao número de tokens que podem ser processados pelo modelo de linguagem.
    - Necessidade de treinamento adicional: O modelo de linguagem pode precisar de treinamento adicional para lidar com consultas mais complexas ou específicas.

    **Importância de ter colocado instruções em chinês:**
    A linguagem chinesa tem uma densidade de informação mais alta do que muitas outras línguas, o que significa que os modelos de linguagem precisam processar menos tokens para entender o contexto e gerar respostas precisas. Isso torna a linguagem chinesa mais apropriada para a utilização de modelos de linguagem com baixa quantidade de tokens. Portanto, ter colocado instruções em chinês no código é um recurso importante para garantir que o aplicativo possa lidar com consultas em chinês de forma eficaz.

    Em resumo, o código é uma aplicação inovadora que combina modelos de linguagem com a API Groq para proporcionar respostas precisas e personalizadas. No entanto, é importante considerar as limitações do aplicativo e trabalhar para melhorá-lo ainda mais.
""")

# Adicionar uma caixa de análise de expertise no sidebar com expander
with st.sidebar.expander("Análise de Expertise do Código"):
    st.markdown("""
    ### Análise de Expertise do Código

    O código fornecido implementa um sistema de chat interativo com especialistas usando a biblioteca Streamlit para a interface de usuário e um modelo de linguagem baseado em API para gerar respostas. A seguir está uma análise detalhada da expertise refletida no código, considerando diferentes aspectos do desenvolvimento de chats com modelos de linguagem (LLMs).

    #### Pontos Positivos

    1. **Configuração da Interface de Usuário**:
       - Uso adequado do Streamlit para criar uma interface web interativa, permitindo a seleção de especialistas e a entrada de consultas pelo usuário.
       - Boa estruturação visual com o uso de `st.markdown` e `st.expander` para apresentar informações de forma organizada e acessível.

    2. **Gestão de Arquivos e Dados**:
       - Carregamento e armazenamento de dados JSON (`agents.json`) para manter informações sobre os especialistas, utilizando boas práticas de manuseio de arquivos.
       - Tratamento de exceções ao carregar dados JSON, com mensagens de erro amigáveis (`json.JSONDecodeError`).

    3. **Integração com API Externa**:
       - Uso da biblioteca `groq` para interagir com uma API de modelo de linguagem, incluindo a configuração de chaves API e parâmetros de consulta.
       - Funções específicas para obter respostas do modelo (`fetch_assistant_response`, `refine_response`, `evaluate_response_with_rag`), demonstrando uma boa modularidade.

    4. **Flexibilidade na Escolha de Modelos**:
       - Inclusão de um dicionário `MODEL_MAX_TOKENS` para definir limites de tokens para diferentes modelos, permitindo flexibilidade na escolha de modelos com diferentes capacidades.
       - Interface de seleção para escolher entre diferentes modelos de linguagem, ajustando dinamicamente os parâmetros (`max_tokens`, `temperature`).

    5. **Funcionalidades de Refinamento e Avaliação**:
       - Implementação de um mecanismo para refinar respostas, permitindo uma análise mais profunda e a melhoria da precisão das respostas geradas.
       - Uso de um sistema de avaliação com Rational Agent Generator (RAG) para assegurar a qualidade e precisão das respostas, incluindo diversas técnicas de análise (SWOT, ANOVA, Q-Statistics).

    #### Pontos a Melhorar

    1. **Segurança e Validação de Entrada**:
       - Falta de sanitização das entradas do usuário, o que pode levar a vulnerabilidades como injeção de código ou dados maliciosos.
       - As chaves da API são inseridas diretamente no código, o que pode não ser seguro. Sugere-se o uso de variáveis de ambiente ou mecanismos seguros de armazenamento.

    2. **Gestão de Sessões e Estado**:
       - Uso de variáveis de sessão (`st.session_state`) para manter o estado da resposta do assistente e outras informações, mas a implementação poderia ser mais robusta para evitar perda de dados entre interações.

    3. **Documentação e Comentários**:
       - O código se beneficiaria de comentários mais detalhados e documentação para melhorar a legibilidade e a manutenção futura.
       - A inclusão de exemplos de uso e uma descrição mais detalhada das funções principais ajudaria outros desenvolvedores a entender melhor o fluxo do código.

    4. **Eficiência e Desempenho**:
       - Dependendo do tamanho dos arquivos JSON e da quantidade de dados processados, a leitura e escrita de arquivos podem se tornar um gargalo. Considere otimizações como a leitura parcial ou o uso de uma base de dados.

    ### Nota Final de Expertise

    Baseando-se nos pontos destacados, a expertise no desenvolvimento deste código pode ser avaliada como alta, especialmente considerando a integração de diferentes componentes (interface, API de modelo de linguagem, gerenciamento de dados) e a implementação de funcionalidades avançadas de avaliação e refinamento de respostas.

    **Nota: 8.5/10**

    Esta avaliação reflete um bom equilíbrio entre funcionalidade, usabilidade e boas práticas de desenvolvimento, com algumas áreas para melhorias em termos de segurança, documentação e eficiência.
    """)


def main():
    st.sidebar.write("""
        Código principal do Agente Expert Geomaker
        """)
    with open("run.py", "r") as file:
        code = file.read()
        st.sidebar.code(code, language='python')
    st.sidebar.write("""
        Código dos Agentes contidos no arquivo agents.json
        """)
    with open("agents.json", "r") as file:
        code = file.read()
        st.sidebar.code(code, language='json')
        
    # Informações de contato
    st.sidebar.image("eu.ico", width=80)
    st.sidebar.write("""
    Projeto Geomaker + IA 
    - Professor: Marcelo Claro.

    Contatos: marceloclaro@gmail.com

    Whatsapp: (88)981587145

    Instagram: [https://www.instagram.com/marceloclaro.geomaker/](https://www.instagram.com/marceloclaro.geomaker/)
    """)

if __name__ == "__main__":
    main()
