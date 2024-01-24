import os
import tempfile
import pdfplumber
import streamlit as st
from streamlit_chat import message
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain.chains import ConversationalRetrievalChain
from pdf2image import convert_from_path
from langchain_community.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings 
from langchain_community.llms import OpenAI

from langchain_community.vectorstores import FAISS
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain



# Access the API key from st.secrets 
openai_api_key = st.secrets["openai_api_key"]
OPENAI_API_KEY = st.secrets["openai_api_key"]
st.set_page_config(page_title="MTI FieldTech AI", layout="wide")

st.image("Images/mti.png", caption='MTI Fieldtech Chatbot', width=300)
class Agent:
        def __init__(self, openai_api_key: str | None = None) -> None:
            # if openai_api_key is None, then it will look the enviroment variable OPENAI_API_KEY
            self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
            self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

            self.llm = OpenAI(temperature=0.3, openai_api_key="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")

            self.chat_history = None
            self.chain = None
            self.db = None

        def ask(self, question: str) -> str:
            if self.chain is None:
                response = "Please, add a document."
            else:
                response = self.chain({"question": question, "chat_history": self.chat_history})
                response = response["answer"].strip()
                self.chat_history.append((question, response))
            return response

        def ingest(self, file_path: os.PathLike) -> None:
      
            try:
                    loader = PyPDFLoader(file_path)
                    documents = loader.load()
                    splitted_documents = self.text_splitter.split_documents(documents)
            except Exception as e:
                    print(f"Error during PDF ingestion: {e}")

                    documents = loader.load()
                    splitted_documents = self.text_splitter.split_documents(documents)

            if self.db is None:
                    self.db = FAISS.from_documents(splitted_documents, self.embeddings)
                    self.chain = ConversationalRetrievalChain.from_llm(self.llm, self.db.as_retriever())
                    self.chat_history = []
            else:
                    self.db.add_documents(splitted_documents)

        def forget(self) -> None:
            self.db = None
            self.chain = None
            self.chat_history = None


with st.sidebar:
    with st.expander("Settings",  expanded=True):
        TEMP = st.slider(label="LLM Temperature", min_value=0.0, max_value=1.0, value=0.3)
        st.markdown("Adjust the LLM Temperature: A higher value makes the output more random, while a lower value makes it more deterministic.")
        st.markdown("NOTE: Anything above 0.7 may produce hallucinations")



        def process_input():
            with st.sidebar:
                if st.session_state["user_input"] and len(st.session_state["user_input"].strip()) > 0:
                    user_text = st.session_state["user_input"].strip()
                    with st.session_state["thinking_spinner"], st.spinner(f"Thinking"):
                        agent_text = st.session_state["agent"].ask(user_text)
                    st.session_state["messages"].append((user_text, True))
                    st.session_state["messages"].append((agent_text, False))
                
            
            




    def is_openai_api_key_set() -> bool:
            return len(st.session_state["OPENAI_API_KEY"]) > 0
    
    def main():
        st.session_state.OPENAI_API_KEY = "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"
        if len(st.session_state) == 0:
                st.session_state["messages"] = []
                st.session_state["OPENAI_API_KEY"] = "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"
        if is_openai_api_key_set():
                st.session_state["agent"] = Agent(st.session_state["OPENAI_API_KEY"])
        else:
                st.session_state["agent"] = None
    def is_openai_api_key_set() -> bool:
            return len(st.session_state["OPENAI_API_KEY"]) > 0
    if "input_OPENAI_API_KEY" not in st.session_state:
        st.session_state.input_OPENAI_API_KEY = ""
    def read_and_save_file():
            if st.session_state["agent"] is not None:  # Check if agent is not None
               st.session_state["agent"].forget()  # Call forget method

            st.session_state["messages"] = []
            st.session_state["user_input"] = ""
            for file in st.session_state["file_uploader"]:
                    with tempfile.NamedTemporaryFile(delete=False) as tf:
                        tf.write(file.getbuffer())
                        temp_pdf_path = tf.name
                        st.session_state["temp_pdf_path"] = temp_pdf_path
                        st.session_state["agent"].ingest(temp_pdf_path)  # Ingest the document
                        images = convert_from_path(temp_pdf_path)
                        st.session_state["images"] = images
with st.sidebar:  
    if st.text_input("OpenAI_API_Key", key="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", type="password"):
        if len(st.session_state.input_OPENAI_API_KEY) > 0 and st.session_state.input_OPENAI_API_KEY != st.session_state.OPENAI_API_KEY:
            st.session_state.OPENAI_API_KEY = st.session_state.input_OPENAI_API_KEY
        if st.session_state["agent"] is not None:
                st.warning("Please, upload the files again.")
                st.session_state["messages"] = []
                st.session_state["user_input"] = ""
                st.session_state["agent"] = Agent(st.session_state["OPENAI_API_KEY"])

    with st.sidebar:
                st.subheader("Upload a document")
                st.file_uploader(
                    "Upload document",
                    type=["pdf"],
                    key="file_uploader",
                    on_change=read_and_save_file,
                    label_visibility="collapsed",
                    accept_multiple_files=True)
                
    temp_pdf_path = st.session_state.get("temp_pdf_path", "")  # Retrieve the path from session_state

    def read_and_save_file():
            if st.session_state["agent"] is not None:  # Check if agent is not None
               st.session_state["agent"].forget()  # Call forget method

            st.session_state["messages"] = []
            st.session_state["user_input"] = ""
            for file in st.session_state["file_uploader"]:
                    with tempfile.NamedTemporaryFile(delete=False) as tf:
                        tf.write(file.getbuffer())
                        temp_pdf_path = tf.name
                        st.session_state["temp_pdf_path"] = temp_pdf_path
                        st.session_state["agent"].ingest(temp_pdf_path)  # Ingest the document
                        images = convert_from_path(temp_pdf_path)
                        st.session_state["images"] = images


    if temp_pdf_path:  # Check if the path is not an empty string
        text = ""  # Initialize the text variable
        with pdfplumber.open(temp_pdf_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() 
    
                images = st.session_state.get("images", [])
        if images:
            page_number = st.slider('Select Page:', min_value=1, max_value=len(images), value=1) - 1
            st.image(images[page_number])
            st.write("PDF processed. Ask your questions below.")

            st.session_state["ingestion_spinner"] = st.empty()
    
        def display_messages():
            st.subheader("Chat")
            for i, (msg, is_user) in enumerate(st.session_state["messages"]):
                message(msg, is_user=is_user, key=str(i))
                st.session_state["thinking_spinner"] = st.empty()
        display_messages()
        st.text_input("Message", key="user_input", disabled=not is_openai_api_key_set(), on_change=process_input)
        
        


        st.divider()
        st.markdown("Source code: [Github](https://github.com/sacred-g/chatpdfs)")
        st.markdown("Created by Steven Bouldin")
        st.markdown("Version: 1.1")




if __name__ == "__main__":
    main()