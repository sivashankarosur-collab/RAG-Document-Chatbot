import os
from langchain_chroma import Chroma

from services.embeddings import EmbeddingModel
from config import DB_DIRECTORY


class VectorDB:

    @staticmethod
    def create(documents):
        embeddings = EmbeddingModel.load_embeddings()

        db_path = os.path.abspath(DB_DIRECTORY)

        vector_db = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=db_path
        )

        return vector_db

    @staticmethod
    def load():
        embeddings = EmbeddingModel.load_embeddings()

        db_path = os.path.abspath(DB_DIRECTORY)

        vector_db = Chroma(
            persist_directory=db_path,
            embedding_function=embeddings
        )

        return vector_db