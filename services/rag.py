from typing import List, Any

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from rank_bm25 import BM25Okapi

from prompts.prompt import RAG_PROMPT
from services.llm import LLM
from services.reranker import Reranker


# ============================================================
# Hybrid Retriever + Reranker
# ============================================================

class HybridRetriever(BaseRetriever):

    vector_db: Any
    documents: List[Document]
    top_k: int = 10
    bm25: Any = None
    reranker: Any = None

    def __init__(self, vector_db, documents=None, top_k=10):

        if documents is None:
            raw = vector_db.get()
            documents = [
                Document(page_content=text, metadata=meta or {})
                for text, meta in zip(raw.get("documents", []), raw.get("metadatas", []))
            ] if raw and "documents" in raw else []

        tokenized_docs = [
            doc.page_content.lower().split()
            for doc in documents
        ]

        bm25_model = BM25Okapi(tokenized_docs)

        reranker_model = Reranker()

        super().__init__(
            vector_db=vector_db,
            documents=documents,
            top_k=top_k,
            bm25=bm25_model,
            reranker=reranker_model
        )

    def _get_relevant_documents(self, query: str) -> List[Document]:

        # ----------------------------------------------------
        # Step 1: Vector Search (Top 10)
        # ----------------------------------------------------

        vector_docs = self.vector_db.similarity_search(
            query,
            k=self.top_k
        )

        # ----------------------------------------------------
        # Step 2: Query Normalization
        # ----------------------------------------------------

        clean_query = (
            query.lower()
            .replace("varshs", "varsha")
            .replace("varsh", "varsha")
        )

        query_tokens = clean_query.split()

        # ----------------------------------------------------
        # Step 3: BM25 Keyword Search
        # ----------------------------------------------------

        scores = self.bm25.get_scores(query_tokens)

        boost_words = clean_query.split()

        for i, doc in enumerate(self.documents):

            text = doc.page_content.lower()
            source = str(doc.metadata.get("source", "")).lower()

            # General keyword boost
            for word in boost_words:
                if word in text:
                    scores[i] += 2.0

            # Strong boost for resume-related queries
            if "cgpa" in clean_query and "resume" in source:
                scores[i] += 10.0

            if "technical skill" in clean_query and "resume" in source:
                scores[i] += 10.0

            if "skill" in clean_query and "resume" in source:
                scores[i] += 5.0

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda idx: scores[idx],
            reverse=True
        )[:self.top_k]

        keyword_docs = [
            self.documents[idx]
            for idx in ranked_indices
        ]

        # ----------------------------------------------------
        # Step 4: Merge and Remove Duplicates
        # ----------------------------------------------------

        merged = []
        seen = set()

        for doc in vector_docs + keyword_docs:

            key = (
                doc.page_content[:100],
                doc.metadata.get("source"),
                doc.metadata.get("page")
            )

            if key not in seen:
                seen.add(key)
                merged.append(doc)

        # ----------------------------------------------------
        # Step 5: Rerank and Keep Top 3
        # ----------------------------------------------------

        reranked_docs = self.reranker.rerank(
            query=clean_query,
            documents=merged,
            top_n=3
        )

        return reranked_docs


# ============================================================
# RAG Service
# ============================================================

class RAGService:

    @staticmethod
    def create_chain(vector_db, documents=None, top_k=10):

        retriever = HybridRetriever(
            vector_db=vector_db,
            documents=documents,
            top_k=top_k
        )

        document_chain = create_stuff_documents_chain(
            llm=LLM.load(),
            prompt=RAG_PROMPT
        )

        retrieval_chain = create_retrieval_chain(
            retriever,
            document_chain
        )

        return retrieval_chain

    @staticmethod
    def get_sources(response):

        sources = []

        documents = response.get("context", [])

        for doc in documents:

            metadata = doc.metadata

            source = metadata.get("source", "Unknown PDF")
            page = metadata.get("page", None)

            if page is not None:

                try:

                    page = int(page) + 1
                    source_info = f"{source} — Page {page}"

                except (ValueError, TypeError):

                    source_info = source

            else:

                source_info = source

            if source_info not in sources:
                sources.append(source_info)

        return sources