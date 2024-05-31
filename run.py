import os
import json
import streamlit as st
from typing import Tuple
from crewai import Agent, Task, Crew, Process
from groq import Groq

# Configurações de API
st.set_page_config(layout="wide")

# Define o caminho para o arquivo JSON que contém os agentes.
FILEPATH = "agents.json"

# Define um dicionário que mapeia nomes de modelos para o número máximo de tokens que cada modelo suporta.
MODEL_MAX_TOKENS = {
    'mixtral-8x7b-32768': 32768,
    'llama3-70b-8192': 8192, 
    'llama3-8b-8192': 8192,
    'gemma-7b-it': 8192,
}

# Função para carregar as opções de agentes a partir do arquivo JSON.
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

# Função para obter o número máximo de tokens permitido por um modelo específico.
def get_max_tokens(model_name: str) -> int:
    return MODEL_MAX_TOKENS.get(model_name, 4096)

# Função para recarregar a página do Streamlit.
def refresh_page():
    st.rerun()

# Função para salvar um novo especialista no arquivo JSON.
def save_expert(expert_title: str, expert_description: str):
    with open(FILEPATH, 'r+') as file:
        agents = json.load(file) if os.path.getsize(FILEPATH) > 0 else []
        agents.append({"agente": expert_title, "descricao": expert_description})
        file.seek(0)
        json.dump(agents, file, indent=4)
        file.truncate()

# Função para obter uma resposta do assistente baseado no modelo Groq.
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

        if agent_selection == "Escolher um especialista...":
            phase_one_prompt = (
                "Saída e resposta obrigatória somente traduzido em português brasileiro. "
                "Assuma o papel de um especialista altamente qualificado em engenharia de prompts e com rigor científico. "
                "Por favor, apresente o código Python com suas bibliotecas respectivas em formato 'markdown' e com comentários detalhados e educacionais em cada linha. "
                "Analise cuidadosamente o requisito apresentado, identificando os critérios que definem as características do especialista mais adequado para lidar com a questão. "
                "Primeiramente, é essencial estabelecer um título que melhor reflita a expertise necessária para fornecer uma resposta completa, aprofundada e clara. "
                "Depois de determinado, descreva minuciosamente as principais habilidades e qualificações desse especialista, evitando vieses. "
                "A resposta deve iniciar com o título do especialista, seguido de um ponto final, e então começar com uma descrição clara, educacional e aprofundada, "
                "que apresente suas características e qualificações que o tornam aptos a lidar com a questão proposta: {user_input} e {user_prompt}. "
                "Essa análise detalhada é crucial para garantir que o especialista selecionado possua o conhecimento e a experiência necessários para fornecer uma resposta "
                "completa e satisfatória, com precisão de 10.0, alinhada aos mais altos padrões profissionais, científicos e acadêmicos. "
                "Nos casos que envolvam código e cálculos, apresente em formato 'markdown' e com comentários detalhados em cada linha. "
                "Resposta deve ser obrigatoriamente em português."
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
            f"Saída e resposta obrigatória somente traduzido em português brasileiro. "
            f"Desempenhando o papel de {expert_title}, um especialista amplamente reconhecido e respeitado em seu campo, "
            f"como doutor e expert nessa área, ofereça uma resposta abrangente e profunda, cobrindo a questão de forma clara, detalhada, expandida, "
            f"educacional e concisa: {user_input} e {user_prompt}. "
            f"Considerando minha longa experiência e profundo conhecimento das disciplinas relacionadas, "
            f"é necessário abordar cada aspecto com atenção e rigor científico. "
            f"Portanto, irei delinear os principais elementos a serem considerados e investigados, fornecendo uma análise detalhada e baseada em evidências, "
            f"evitando vieses e citando referências conforme apropriado: {user_prompt}. "
            f"O objetivo final é fornecer uma resposta completa e satisfatória, alinhada aos mais altos padrões acadêmicos e profissionais, "
            f"atendendo às necessidades específicas da questão apresentada. "
            f"Certifique-se de apresentar a resposta em formato 'markdown', com comentários detalhados em cada linha. "
            f"Mantenha o padrão de escrita em 10 parágrafos, cada parágrafo com 4 sentenças, e cada sentença separada por vírgulas, "
            f"seguindo sempre as melhores práticas pedagógicas aristotélicas."
        )
        phase_two_response = get_completion(phase_two_prompt)

    except Exception as e:
        st.error(f"Ocorreu um erro: {e}")
        return "", ""

    return expert_title, phase_two_response

# Função para criar um agente com base no índice
def create_agent(index):
    return Agent(
        role=f'Agente{index}',
        goal=f'Solução do problema proposto pelo usuário',
        verbose=True,
        memory=True,
        backstory=(
            f'Você é o Agente{index}, um especialista altamente qualificado com '
            f'extensa experiência em resolver problemas complexos. Seu objetivo é '
            f'fornecer a melhor solução possível para o problema apresentado.'
        )
    )

# Função para criar uma tarefa para cada agente
def create_task(agent, index):
    return Task(
        description=(
            f'Agente{index}, sua tarefa é analisar o problema fornecido pelo usuário e '
            f'propor uma solução inicial detalhada.'
        ),
        expected_output=f'Solução inicial proposta pelo Agente{index}',
        tools=[],  # Adicione ferramentas conforme necessário
        agent=agent,
    )

# Função para executar as rodadas de debate
def execute_debate(crew, rounds):
    for round in range(1, rounds + 1):
        st.write(f"Iniciando rodada {round}")
        result = crew.kickoff(inputs={'problema': st.session_state["problema_usuario"]})
        st.write(f"Resultado da rodada {round}: {result}")
        st.write("Pressione Enter para continuar para a próxima rodada...")

def main():
    st.image('updating.gif', width=300, caption='Laboratório de Educação e Inteligência Artificial - Geomaker. "A melhor forma de prever o futuro é inventá-lo." - Alan Kay', use_column_width='always', output_format='auto')
    st.markdown("<h1 style='text-align: center;'>Agentes Alan Kay</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center;'>Utilize o Rational Agent Generator (RAG) para avaliar a resposta do especialista e garantir qualidade e precisão.</h2>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    st.write("Digite sua solicitação para que ela seja respondida pelo especialista ideal.")

    user_input = st.text_area("Por favor, insira sua solicitação:", height=200, key="entrada_usuario")
    user_prompt = st.text_area("Escreva um prompt ou coloque o texto para consulta para o especialista (opcional):", height=200, key="prompt_usuario")
    agent_selection = st.selectbox("Escolha um Especialista", options=load_agent_options(), index=0, key="selecao_agente")
    model_name = st.selectbox("Escolha um Modelo", list(MODEL_MAX_TOKENS.keys()), index=0, key="nome_modelo")
    temperature = st.slider("Nível de Criatividade", min_value=0.0, max_value=1.0, value=0.0, step=0.01, key="temperatura")
    groq_api_key = st.text_input("Chave da API Groq: Você pode usar esse como teste - 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 ", key="groq_api_key")

    if st.button("Buscar Resposta"):
        if not user_input:
            st.warning("Por favor, insira sua solicitação.")
        else:
            st.session_state["problema_usuario"] = user_input
            st.session_state["descricao_especialista_ideal"], st.session_state["resposta_assistente"] = fetch_assistant_response(user_input, user_prompt, model_name, temperature, agent_selection, groq_api_key)

            # Configuração dos agentes
            agents = [create_agent(i) for i in range(1, 10)]
            tasks = [create_task(agent, i) for i, agent in enumerate(agents, 1)]

            # Configuração do processo de debate
            crew = Crew(
                agents=agents,
                tasks=tasks,
                process=Process.sequential
            )

            execute_debate(crew, 30)

    if st.button("Apagar"):
        st.session_state.clear()
        st.experimental_rerun()

    st.sidebar.image("logo.png", width=200)
    st.sidebar.write("""
    Projeto Geomaker + IA 
    - Professor: Marcelo Claro.

    Contatos: marceloclaro@gmail.com

    Whatsapp: (88)981587145

    Instagram: [https://www.instagram.com/marceloclaro.geomaker/](https://www.instagram.com/marceloclaro.geomaker/)
    """)


