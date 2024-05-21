from io import BytesIO
import os
from docx import Document
from dotenv import load_dotenv
import streamlit as st
from streamlit_chat import message
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain.chains import ConversationChain
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from redundant_filter_retriever import RedundantFilterRetriever
import asyncio
import pyperclip

def get_or_create_eventloop():
    try:
        return asyncio.get_event_loop()
    except RuntimeError as ex:
        if "There is no current event loop in thread" in str(ex):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return asyncio.get_event_loop()

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)
load_dotenv()

st.set_page_config(
    page_title="MeitY Guidelines BOT",
    page_icon="🤖",
    layout="wide"
)

# Sidebar
st.sidebar.title("Ministry of Electronics and Information Technology")
st.sidebar.markdown("Welcome to MietY Guidelines QnA!")
st.sidebar.markdown("Here are some useful links:")
st.sidebar.markdown("[MeitY Homepage](https://www.meity.gov.in)")
st.sidebar.markdown("[Guidelines](https://www.meity.gov.in/policies-guidelines)")
st.sidebar.markdown("[FAQs](https://www.meity.gov.in/frequently-asked-questions)")

if st.sidebar.button("Start New Chat"):
    st.session_state['responses'] = ["How can I assist you?"]
    st.session_state['requests'] = []
    st.session_state['metadata'] = []
    st.session_state.buffer_memory.clear()

st.title("MeitY Chatbot")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", google_api_key=GOOGLE_API_KEY)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

db = Chroma(
    persist_directory="emb",
    embedding_function=embeddings
)

retriever = RedundantFilterRetriever(
    embeddings=embeddings,
    chroma=db
)

if 'responses' not in st.session_state:
    st.session_state['responses'] = ["How can I assist you?"]

if 'requests' not in st.session_state:
    st.session_state['requests'] = []

if 'buffer_memory' not in st.session_state:
    st.session_state.buffer_memory = ConversationBufferWindowMemory(k=3, return_messages=True, memory_key="history")

if 'metadata' not in st.session_state:
    st.session_state['metadata'] = []

if "copied" not in st.session_state: 
    st.session_state.copied = []

system_msg_template = SystemMessagePromptTemplate.from_template(
    template="""Answer the question as relevant as possible using the provided context on MIETY guidelines implementing services, 
    don't use internet and try to provide as much text as possible from the "response" section in the source document context 
    without making many changes being helpful. Ignore the upper and lower case difference in query from context & If the answer is not contained within the context, only then say 'I don't know' or else just be as helpful as possible."""
)

human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")

prompt_template = ChatPromptTemplate.from_messages([
    system_msg_template,
    MessagesPlaceholder(variable_name="history"),
    human_msg_template
])

conversation = ConversationChain(
    memory=st.session_state.buffer_memory,
    prompt=prompt_template,
    llm=llm,
    verbose=False
)

response_container = st.container()
textcontainer = st.container()

def get_conversation_string():
    conversation_string = ""
    for i in range(len(st.session_state['responses']) - 1):
        conversation_string += "Human: " + st.session_state['requests'][i] + "\n"
        conversation_string += "Bot: " + st.session_state['responses'][i + 1] + "\n"
    return conversation_string

model = GoogleGenerativeAI(model="gemini-pro")

def query_refiner(conversation, query):
    response = model.invoke(f"Given the following user query and conversation log, formulate a question that would be the most relevant to provide the user with an answer from a knowledge base.\n\nCONVERSATION LOG: \n{conversation}\n\nQuery: {query}\n\nRefined Query:")
    return response

query = st.chat_input("How can I guide you!", key="input")
with textcontainer:
    if query:
        conversation_string = get_conversation_string()
        refined_query = query_refiner(conversation_string, query)
        context_docs = db.similarity_search_with_score(refined_query, k=3)
        best_source = None
        best_page = None
        best_score = -1
        context_texts = ""
        
        for doc in context_docs:
            context_texts += doc[0].page_content + "\n" 
            source = doc[0].metadata['source']
            page = doc[0].metadata['page']
            score = doc[0].metadata.get('similarity_score', 0)  
            
            if score > best_score:
                best_source = source
                best_page = page
                best_score = score
        
        best_source_text = f"Source: {best_source}, Page: {best_page}"
        response = conversation.predict(input=f"Context:\n {context_texts} \n\n Query:\n {query}")

        st.session_state.requests.append(query)
        st.session_state.responses.append(response)
        st.session_state.metadata.append(best_source_text)
def copy_to_clipboard(response):
    pyperclip.copy(response)

with response_container:
    if st.session_state['responses']:
        for i in range(len(st.session_state.responses)):
            message(st.session_state['responses'][i], key=str(i))
            if i < len(st.session_state['requests']):
                message(st.session_state["requests"][i], is_user=True, key=str(i) + '_user')
                
                metadata_col, button_col = st.columns([3, 1])
                with metadata_col:
                    st.markdown(f"{st.session_state['metadata'][i]}")
                with button_col:
                    st.button("📋", on_click=copy_to_clipboard(st.session_state['responses'][i+1]),help="Copy response to clipboard",)

def export_to_word():
    doc = Document()
    doc.add_heading('Chat History', level=1)
    for i in range(len(st.session_state['responses']) - 1):
        doc.add_heading(f'Q{i + 1}:', level=2)
        doc.add_paragraph(st.session_state['requests'][i])
        doc.add_heading(f'A{i + 1}:', level=2)
        doc.add_paragraph(st.session_state['responses'][i + 1])
        if i < len(st.session_state['metadata']):
            doc.add_paragraph(f'Metadata: {st.session_state["metadata"][i]}')
    
    buffer = BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    
    st.sidebar.download_button(
        label="Export Chat as Word Document",
        data=buffer,
        file_name="MeitY_guidelines_chat_history.docx",
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )

st.sidebar.title("Actions")
export_to_word()