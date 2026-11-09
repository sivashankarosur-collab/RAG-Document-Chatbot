from langchain_groq import ChatGroq
from config import GROQ_MODEL
import os
from dotenv import load_dotenv

load_dotenv()


class LLM:

    @staticmethod
    def load():

        llm = ChatGroq(
            model=GROQ_MODEL,
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0
        )

        return llm