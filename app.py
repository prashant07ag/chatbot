import os
from dotenv import load_dotenv
import streamlit as st
from streamlit_chat import message
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain.chains import ConversationChain, RetrievalQA
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
# Load environment variables
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
    st.session_state.buffer_memory = ConversationBufferWindowMemory(k=3,
     return_messages=True,memory_key="history")

system_msg_template = SystemMessagePromptTemplate.from_template(
    template="""Answer the question as relevant as possible using the provided context on MIETY guidelines implementing services, 
    don't use internet and try to provide as much text as possible from the "response" section in the source document context 
    without making many changes being helpful. Ignore the upper and lower case diffrence in query from context & If the answer is not contained within the context,  only than say 'I don't know or else just be as helpfull assistant."""
)

human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")

prompt_template = ChatPromptTemplate.from_messages([
    system_msg_template,
    MessagesPlaceholder(variable_name="history"),
    human_msg_template
])

# context_chain = RetrievalQA.from_chain_type(
#         llm=llm,
#         retriever=retriever,
#         chain_type="stuff")

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
        context = db.similarity_search(refined_query, k=3)
        
        context_texts = "\n".join([doc.page_content for doc in context])
        response = conversation.predict(input=f"Context:\n {context_texts} \n\n Query:\n {refined_query}")

        st.session_state.requests.append(query)
        st.session_state.responses.append(response)

with response_container:
    if st.session_state['responses']:
        for i in range(len(st.session_state.responses)):
            message(st.session_state['responses'][i], key=str(i))
            if i < len(st.session_state['requests']):
                message(st.session_state["requests"][i], is_user=True, key=str(i) + '_user')
