from services.vector_db import VectorDB

db = VectorDB.load()

question = "What is the objective of the project?"

results = db.similarity_search_with_score(question, k=10)

print("=" * 80)
print("QUESTION:", question)
print("=" * 80)

for i, (doc, score) in enumerate(results):
    print("\n" + "=" * 80)
    print("RESULT:", i + 1)
    print("SCORE:", score)
    print("CONTENT:")
    print(doc.page_content[:1200])