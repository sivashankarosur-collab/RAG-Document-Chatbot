from sentence_transformers import SentenceTransformer, util


class Reranker:

    def __init__(self):

        # Lightweight and fast reranker model
        self.model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

    def rerank(self, query, documents, top_n=3):

        if not documents:
            return []

        # Document texts
        doc_texts = [
            doc.page_content
            for doc in documents
        ]

        # Encode query and documents
        query_embedding = self.model.encode(
            query,
            convert_to_tensor=True
        )

        doc_embeddings = self.model.encode(
            doc_texts,
            convert_to_tensor=True
        )

        # Similarity scores
        scores = util.cos_sim(
            query_embedding,
            doc_embeddings
        )[0]

        # Sort by score
        ranked_indices = scores.argsort(
            descending=True
        )[:top_n]

        reranked_docs = [
            documents[int(i)]
            for i in ranked_indices
        ]

        return reranked_docs