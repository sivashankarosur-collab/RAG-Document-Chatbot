from langchain_core.prompts import ChatPromptTemplate

RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful assistant that answers questions from a PDF.

Use ONLY the context provided below.

If the answer is present in the context, answer directly.

Never say 'There is no information provided' unless the context is truly empty.

Context:
{context}
""",
        ),
        ("human", "{input}"),
    ]
)