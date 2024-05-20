from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import warnings
from langchain_experimental.text_splitter import SemanticChunker
load_dotenv()
warnings.filterwarnings("ignore", category=DeprecationWarning)
import PyPDF2

# def get_similiar_docs(query,k=1,score=False):
#   if score:
#     similar_docs = db.similarity_search_with_score(query,k=k)
#   else:
#     similar_docs = db.similarity_search(query,k=k)
#   return similar_docs

# def extract_text_from_pdf(pdf_path):
#     text = ""
#     with open(pdf_path, "rb") as file:
#         pdf_reader = PyPDF2.PdfReader(file)
#         for page_number in range(len(pdf_reader.pages)):
#             page = pdf_reader.pages[page_number]
#             text += page.extract_text()
#     return text

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 2500,
    chunk_overlap = 250,
    length_function=len)

loader = PyPDFLoader("MIeTY guideline implementing-services.pdf")
docs = loader.load_and_split(text_splitter=text_splitter)
print(len(docs))

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    # Create Chroma index from documents
db = Chroma.from_documents(
        docs,
        embedding=embeddings,
        persist_directory="emb",
    )

