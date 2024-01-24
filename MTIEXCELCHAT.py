import io
import logging
import uuid
from pathlib import Path
from typing import Dict
from util import get_openai_model, get_ollama_model, get_baidu_as_model, get_prompt_template, get_gemini_model
import matplotlib
import pandas as pd
import streamlit as st
from pandasai.config import Config  # Assuming Config is imported from the correct module
from pandasai import Agent, SmartDataframe, Config
from pandasai.helpers import Logger
from parser.response_parser import CustomResponseParser
from middleware.base import CustomChartsMiddleware
from st_aggrid import AgGrid
logger = Logger()

import toml

# Access the API key using the os.getenv function


# Use the API key in your code

api_token = st.secrets["api_token"]


matplotlib.rc_file("./.matplotlib/.matplotlibrc")

# page settings
st.set_page_config(page_title="Home", layout="wide")
st.image('docs/images/logotag_white_col.png')
st.header("Welcome to ChatExcel! 📊✨")
st.text('''ChatExcel is a powerful and intuitive chat application designed to enhance your interaction with Excel data. Seamlessly converse with your data, ask questions, request edits,
        "and witness your spreadsheet transform dynamically based on your queries.''')
st.markdown("""Key Features
 - [1] Excel Upload: Easily upload your Excel files to start the conversation. Our app supports various formats, ensuring a smooth experience for your data-driven discussions.
 - [2] Natural Language Interaction: Communicate with your data using natural language. Ask questions, seek insights, and get responses in a conversational format. No need to be an Excel expert!
 - [3] Data Editing: Take control of your data by requesting edits directly through the chat interface. Specify changes, and watch as your Excel sheet transforms based on your instructions.
 - [4] Download Edited Data: Once you're satisfied with the edits, simply click a button to download the modified version of your Excel file. It's that easy!
""")
st.sidebar.success ("Select a app from above" )
class AgentWrapper:
    id: str
    agent: Agent

    def __init__(self) -> None:
        self.agent = None
        self.id = str(uuid.uuid4())

    def get_llm(self):
        op = st.session_state.last_option
        llm = None
        if op == "Ollama":
            llm = get_ollama_model(st.session_state.ollama_model, st.session_state.ollama_base_url)
        elif op == "OpenAI":
            openai_secrets = toml.load("secrets.toml").get("openai", {})
            api_token = openai_secrets.get("api_token", "")
            
            if api_token != "":
                llm = get_openai_model(api_token)
        elif op == "Baidu/AIStudio-Ernie-Bot":
            if st.session_state.access_token != "":
                llm = get_baidu_as_model(st.session_state.access_token)
        elif op == "Baidu/Qianfan-Ernie-Bot":
            if st.session_state.client_id != "" and st.session_state.client_secret != "":
                llm = get_baidu_qianfan_model(st.session_state.client_id, st.session_state.client_secret)
        if llm is None:
            st.toast("LLM initialization failed, check LLM configuration", icon="🫤")
        return llm

    def set_file_data(self, df):
        llm = self.get_llm()
        if llm is not None:
            print("llm.type", llm.type)
            config = Config(
                llm=llm,
                middlewares=[CustomChartsMiddleware()],
                response_parser=CustomResponseParser,
                custom_prompts={
                    "generate_python_code": get_prompt_template()
                },
                enable_cache=False,
                verbose=True
            )
            self.agent = Agent(df, config=config, memory_size=memory_size)
            self.agent._lake.add_middlewares(CustomChartsMiddleware())
            st.session_state.llm_ready = True

    def chat(self, prompt):
        if self.agent is None:
            st.toast("LLM initialization failed, check LLM configuration", icon="🫣")
            st.stop()
        else:
            return self.agent.chat(prompt)

    def start_new_conversation(self):
        self.agent.start_new_conversation()
        st.session_state.chat_history = []


@st.cache_resource
def get_agent(agent_id) -> AgentWrapper:
    agent = AgentWrapper()
    return agent

chat_history_key = "chat_history"
if chat_history_key not in st.session_state:
    st.session_state[chat_history_key] = []


if "llm_ready" not in st.session_state:
    st.session_state.llm_ready = False

# Description
tab1, tab2, tab3 = st.tabs(["Workspace", "Screenshots","Playground"])
with tab2:
    col1, col2 = st.columns(2)
    with col1:
        st.image("docs/images/short1.jpeg")
        st.image("docs/images/short1.png")
    with col2:
        st.image("docs/images/short2.png")
        
with tab3:
    col1, col2 = st.columns(2)
    with col1:
         df = pd.DataFrame(
    [
       {"command": "st.selectbox", "rating": 4, "is_widget": True},
       {"command": "st.balloons", "rating": 5, "is_widget": False},
       {"command": "st.time_input", "rating": 3, "is_widget": True},
   ]
)

    edited_df = st.data_editor(df, num_rows="dynamic")
    with col2:
            data_df = pd.DataFrame(
        {
            "sales": [
                [0, 4, 26, 80, 100, 40],
                [80, 20, 80, 35, 40, 100],
                [10, 20, 80, 80, 70, 0],
                [10, 100, 20, 100, 30, 100],
            ],
        }
    )



# DataGrid
with st.expander("DataGrid Content") as ep:
    grid = st.dataframe(pd.DataFrame(), use_container_width=True)
counter = st.markdown("")

# Sidebar layout
with st.sidebar:
    option = st.selectbox("Choose LLM", ["OpenAI", "Baidu/AIStudio-Ernie-Bot", "Baidu/Qianfan-Ernie-Bot", "Ollama"])

    # Initialize session keys
    if "api_token" not in st.session_state:
        st.session_state.api_token = ""
    if "access_token" not in st.session_state:
        st.session_state.access_token = ""
    if "ollama_model" not in st.session_state:
        st.session_state.ollama_model = ""
    if "ollama_base_url" not in st.session_state:
        st.session_state.ollama_base_url = ""
    if "client_id" not in st.session_state:
        st.session_state.client_id = ""
    if "client_secret" not in st.session_state:
        st.session_state.client_secret = ""

    # Initialize model configration panel
    if option == "OpenAI":
        api_token = st.text_input("API Token", st.session_state.api_token, type="password", placeholder="Api token")
    elif option == "Baidu/AIStudio-Ernie-Bot":
        access_token = st.text_input("Access Token", st.session_state.access_token, type="password",
                                     placeholder="Access token")
    elif option == "Baidu/Qianfan-Ernie-Bot":
        client_id = st.text_input("Client ID", st.session_state.client_id, placeholder="Client ID")
        client_secret = st.text_input("Client Secret", st.session_state.client_secret, type="password",
                                      placeholder="Client Secret")
    elif option == "Ollama":
        ollama_model = st.selectbox(
            "Choose Ollama Model",
            ["starcoder:7b", "codellama:7b-instruct-q8_0", "zephyr:7b-alpha-q8_0"]
        )
        ollama_base_url = st.text_input("Ollama BaseURL", st.session_state.ollama_base_url,
                                        placeholder="http://localhost:11434")

    memory_size = st.selectbox("Memory Size", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], index=9)

    if st.button("+ New Chat"):
        st.session_state.llm_ready = False
        st.session_state[chat_history_key] = []

    # Validation
    info = st.markdown("")
    if option == "OpenAI":
        if not api_token:
            info.error("Invalid API Token")
        if api_token != st.session_state.api_token:
            st.session_state.api_token = api_token
            st.session_state.llm_ready = False
    elif option == "Baidu/AIStudio-Ernie-Bot":
        if not access_token:
            info.error("Invalid Access Token")
        if access_token != st.session_state.access_token:
            st.session_state.access_token = access_token
            st.session_state.llm_ready = False
    elif option == "Baidu/Qianfan-Ernie-Bot":
        if client_id != st.session_state.client_id:
            st.session_state.client_id = client_id
            st.session_state.llm_ready = False
        if client_secret != st.session_state.client_secret:
            st.session_state.client_secret = client_secret
            st.session_state.llm_ready = False
    elif option == "Ollama":
        if ollama_model != st.session_state.ollama_model:
            st.session_state.ollama_model = ollama_model
            st.session_state.llm_ready = False
        if ollama_base_url != st.session_state.ollama_base_url:
            st.session_state.ollama_base_url = ollama_base_url
            st.session_state.llm_ready = False

    if "last_option" not in st.session_state:
        st.session_state.last_option = None

    if option != st.session_state.last_option:
        st.session_state.last_option = option
        st.session_state.llm_ready = False

    if "last_memory_size" not in st.session_state:
        st.session_state.last_memory_size = None

    if memory_size != st.session_state.last_memory_size:
        st.session_state.last_memory_size = memory_size
        st.session_state.llm_ready = False

logger.log(f"st.session_state.llm_ready={st.session_state.llm_ready}", level=logging.INFO)

if not st.session_state.llm_ready:
    st.session_state.agent_id = str(uuid.uuid4())

with st.sidebar:
    st.divider()
    file = st.file_uploader("Upload File", type=["xlsx", "csv"])
    if file is None:
        st.session_state.uploaded = False
        if st.session_state.llm_ready:
            get_agent(st.session_state.agent_id).start_new_conversation()

    if "last_file" not in st.session_state:
        st.session_state.last_file = None

    if file is not None:
        file_obj = io.BytesIO(file.getvalue())
        file_ext = Path(file.name).suffix.lower()
        if file_ext == ".csv":
            df = pd.read_csv(file_obj)
        else:
            df = pd.read_excel(file_obj)
        grid.dataframe(df)
        counter.info("Total: **%s** records" % len(df))

        if file != st.session_state.last_file or st.session_state.llm_ready is False:
            # if not st.session_state.llm_ready:
            st.session_state.agent_id = str(uuid.uuid4())
            get_agent(st.session_state.agent_id).set_file_data(df)

        st.session_state.last_file = file

with st.sidebar:
    st.markdown("""
    <style>
        .tw_share {
            # position: fixed;
            display: inline-block;
            # left: 240px;
            # bottom: 20px;
            cursor: pointer;
        }
        
        .tw_share a {
            text-decoration: none;
        }
        
        .tw_share span {
            color: white;
        }
        
        .tw_share span {
            margin-left: 2px;
        }
        
        .tw_share:hover svg path {
            fill: #1da1f2;
        }
        
        .tw_share:hover span {
            color: #1da1f2;
        }
    </style>
    <div class="tw_share">
        <a target="_blank" href=<span>Created By: Steven Bouldin</span></a>
    </div>
    """, unsafe_allow_html=True)

# ChatBox layout


            

for item in st.session_state.chat_history:
    with st.chat_message(item["role"]):
        if "type" in item and item["type"] == "plot":
            tmp = st.image(item['content'])
        elif "type" in item and item["type"] == "dataframe":
            tmp = st.dataframe(item['content'])
        else:
            st.markdown(item["content"])

prompt = st.chat_input("Input the question here")
if prompt is not None:
    st.chat_message("user").markdown(prompt)
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("assistant"):
        if not st.session_state.llm_ready:
            response = "Please upload the file and configure the LLM first"
            st.markdown(response)
            st.session_state.chat_history.append({"role": "assistant", "content": response})
        else:
            tmp = st.markdown(f"Analyzing, hold on pls...")
            
        response = get_agent(st.session_state.agent_id).chat(prompt)

        if isinstance(response, SmartDataframe):
            tmp.dataframe(response.dataframe)
            st.session_state.chat_history.append(
                {"role": "assistant", "content": response.dataframe, "type": "dataframe"})
        elif isinstance(response, Dict) and "type" in response and response["type"] == "plot":
            tmp.image(f"{response['value']}")
            st.session_state.chat_history.append(
                {"role": "assistant", "content": response["value"], "type": "plot"}),
   
            st.session_state.chat_history.append({"role": "assistant", "content": response["value"], "type": "plot"})
        else:
            AgGrid(df)
            tmp.markdown(response)
            st.session_state.chat_history.append({"role": "assistant", "content": response})

        # The following lines should be indented further to align with the code block under the 'else' statement
        # if instance(dict, file_name=temp.png)
        #directory_save, mtiexcel_chat
        
        if isinstance(response, SmartDataframe):
            csv = df.to_csv(index=False)
            st.download_button(
            label='Download Edited Data',
            data=csv,
            file_name='edited_data.csv',
            mime='text/csv/xlsx'
            )
      
                
                
