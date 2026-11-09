import os

from loaders.pdf_loader import PDFLoader
from services.vector_db import VectorDB
from langchain_text_splitters import RecursiveCharacterTextSplitter


pdf_path = os.path.join(os.path.dirname(__file__), "data", "application hclt.pdf")

docs = PDFLoader.load_pdf(pdf_path)

print("Pages loaded:", len(docs))

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=300
)

chunks = text_splitter.split_documents(docs)

print("Chunks created:", len(chunks))

db = VectorDB.create(chunks)

print("Database Created Successfully")
