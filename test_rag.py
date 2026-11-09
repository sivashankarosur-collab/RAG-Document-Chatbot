from services.vector_db import VectorDB
from services.rag import RAGService

print("Loading database...")
db = VectorDB.load()

print("Creating RAG chain...")
chain = RAGService.create_chain(db)

question = "What is the objective of the project?"

print("\nQuestion:", question)
print("\nCalling LLM...")

try:
    response = chain.invoke({
        "input": question
    })

    print("\n" + "=" * 70)
    print("ANSWER")
    print("=" * 70)
    print(response.get("answer", "NO ANSWER FOUND"))

    print("\n" + "=" * 70)
    print("RETRIEVED CONTEXT")
    print("=" * 70)

    for i, doc in enumerate(response.get("context", [])):
        print(f"\n--- DOCUMENT {i + 1} ---")
        print(doc.page_content[:1500])

except Exception as e:
    print("\nERROR:")
    print(type(e).__name__)
    print(str(e))