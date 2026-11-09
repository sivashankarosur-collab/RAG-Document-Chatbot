import os

from loaders.pdf_loader import PDFLoader

from vectorstore.vector_store import create_vectorstore

from config import DB_PATH


def initialize_database():

    if not os.path.exists(DB_PATH):

        docs = PDFLoader.load_pdf(os.path.join(os.path.dirname(__file__), "..", "data", "siva__resume.pdf"))

        create_vectorstore(docs)
