from src.helper import load_pdf_file, text_splitter, download_hugging_face_embeddings
from pinecone import ServerlessSpec
from dotenv import load_dotenv
from langchain_community.vectorstores import Pinecone as PineconeVectorStore
import os
from pinecone.grpc import PineconeGRPC as pinecone


load_dotenv()
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

extracted_data = load_pdf_file(data="data/")
text_chunks = text_splitter(extracted_data)
embeddings = download_hugging_face_embeddings()

docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name="rag",
    embedding=embeddings 
   
)
