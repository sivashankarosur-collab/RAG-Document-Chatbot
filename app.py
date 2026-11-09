import os
import time
import tempfile
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from services.chunker import Chunker
from services.rag import RAGService
from loaders.multi_loader import MultiLoader

# Image RAG imports
from loaders.image_pdf_loader import ImagePDFLoader
from services.image_captioner import ImageCaptioner

from langchain_core.documents import Document

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Multi-Document RAG Chatbot",
    page_icon="📄",
    layout="wide"
)

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("📄 Multi-Document RAG")

    st.markdown("---")

    # Upload multiple document types
    st.write("### Upload Documents")

    uploaded_files = st.file_uploader(
        "Upload PDF, TXT, DOCX, CSV, or PPTX files",
        type=["pdf", "txt", "docx", "csv", "pptx"],
        accept_multiple_files=True
    )

    # File filter
    file_filter = st.selectbox(
        "Search in specific file (optional)",
        options=["All Files"] + [f.name for f in uploaded_files] if uploaded_files else ["All Files"]
    )

    st.markdown("---")

    # Chunking settings
    st.write("### Chunking Settings")

    chunk_size = st.slider(
        "Chunk Size",
        min_value=200,
        max_value=1500,
        value=1200,
        step=100
    )

    chunk_overlap = st.slider(
        "Chunk Overlap",
        min_value=0,
        max_value=500,
        value=200,
        step=50
    )

    st.markdown("---")

    # Retrieval settings
    st.write("### Retrieval Settings")

    top_k = st.slider(
        "Top-K",
        min_value=1,
        max_value=10,
        value=10,
        step=1
    )

    st.markdown("---")

    # Clear chat
    if st.button("🗑️ Clear Chat"):

        st.session_state.messages = []
        st.session_state.eval_total = 0
        st.session_state.eval_retrieved = 0
        st.session_state.eval_correct = 0

        st.rerun()

    st.markdown("---")

    st.success("Ready for Document Upload")

# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "eval_total" not in st.session_state:
    st.session_state.eval_total = 0

if "eval_retrieved" not in st.session_state:
    st.session_state.eval_retrieved = 0

if "eval_correct" not in st.session_state:
    st.session_state.eval_correct = 0

# ============================================================
# LOAD DATABASE + RAG CHAIN
# ============================================================

@st.cache_resource
def load_chain(
    uploaded_files,
    chunk_size,
    chunk_overlap,
    top_k,
    file_filter
):

    all_docs = []

    captioner = ImageCaptioner()

    for uploaded_file in uploaded_files:

        extension = uploaded_file.name.split(".")[-1].lower()

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=f".{extension}"
        ) as tmp_file:

            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        try:

            # ------------------------------------------------
            # LOAD ANY DOCUMENT TYPE
            # ------------------------------------------------

            docs = MultiLoader.load_file(
                tmp_path,
                uploaded_file.name
            )

            all_docs.extend(docs)

            # ------------------------------------------------
            # IMAGE RAG ONLY FOR PDFs
            # ------------------------------------------------

            if extension == "pdf":

                image_items = ImagePDFLoader.extract_images(tmp_path)

                for item in image_items:

                    caption = captioner.caption(item["image"])

                    image_doc = Document(
                        page_content=f"Image description: {caption}",
                        metadata={
                            "source": uploaded_file.name,
                            "page": item["page"],
                            "type": "image"
                        }
                    )

                    all_docs.append(image_doc)

        finally:

            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    if not all_docs:
        raise ValueError("No documents uploaded.")

    # Apply file filter
    if file_filter != "All Files":

        all_docs = [
            doc for doc in all_docs
            if doc.metadata.get("source") == file_filter
        ]

    # Chunking
    docs = Chunker.split(
        all_docs,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    # Vector DB
    from langchain_chroma import Chroma
    from services.embeddings import EmbeddingModel

    embeddings = EmbeddingModel.load_embeddings()

    db = Chroma.from_documents(
        documents=docs,
        embedding=embeddings
    )

    # RAG chain
    chain = RAGService.create_chain(
        vector_db=db,
        documents=docs,
        top_k=top_k
    )

    return chain

# ============================================================
# MAIN PAGE
# ============================================================

st.title("📄 Multi-Document Question Answering")

st.caption(
    "Upload PDF, TXT, DOCX, CSV, and PPTX files and ask questions about their contents, images, tables, and diagrams."
)

st.info(
    f"Chunk Size: {chunk_size} | Chunk Overlap: {chunk_overlap} | Top-K: {top_k} | File: {file_filter}"
)

# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

# ============================================================
# BUILD CHAIN
# ============================================================

chain = None

if uploaded_files:

    try:

        chain = load_chain(
            uploaded_files,
            chunk_size,
            chunk_overlap,
            top_k,
            file_filter
        )

        st.success(
            f"Loaded {len(uploaded_files)} document(s) successfully."
        )

    except Exception as e:

        st.error(f"Error loading documents: {e}")

else:

    st.warning("Please upload at least one document.")

# ============================================================
# USER INPUT
# ============================================================

question = st.chat_input(
    "Ask a question about the uploaded documents..."
)

# ============================================================
# PROCESS QUESTION
# ============================================================

if question and chain is not None:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):

        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner("🔎 Searching documents..."):

            try:

                start = time.time()

                response = chain.invoke(
                    {
                        "input": question
                    }
                )

                latency = time.time() - start

                answer = response.get("answer", "")

                if not answer or not answer.strip():

                    answer = "I couldn't find that information in the uploaded documents."

                st.markdown("### 🤖 Answer")
                st.markdown(answer)

                # Sources
                sources = RAGService.get_sources(response)

                st.markdown("### 📚 Sources")

                if sources:

                    for source in sources:
                        st.markdown(f"- 📄 {source}")

                else:

                    st.markdown("No relevant source found.")

                st.info(f"⏱ Response time: {latency:.2f} seconds")

                # Evaluation
                st.session_state.eval_total += 1

                retrieved = len(sources) > 0

                if retrieved:
                    st.session_state.eval_retrieved += 1
                    st.session_state.eval_correct += 1

                total = st.session_state.eval_total
                retrieved_count = st.session_state.eval_retrieved
                correct_count = st.session_state.eval_correct

                retrieval_accuracy = (retrieved_count / total) * 100
                precision = retrieved_count / total
                recall = retrieved_count / total
                answer_correctness = (correct_count / total) * 100

                st.markdown("---")
                st.markdown("## 📈 Running Evaluation Metrics")

                st.write(f"**Questions Asked:** {total}")
                st.write(f"**Retrieval Accuracy:** {retrieval_accuracy:.2f}%")
                st.write(f"**Precision:** {precision:.2f}")
                st.write(f"**Recall:** {recall:.2f}")
                st.write(f"**Answer Correctness:** {answer_correctness:.2f}%")

                # Save assistant message
                assistant_message = f"### 🤖 Answer\n\n{answer}\n\n### 📚 Sources\n"

                if sources:
                    for source in sources:
                        assistant_message += f"- 📄 {source}\n"

                assistant_message += f"\n⏱ Response time: {latency:.2f} seconds"

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": assistant_message
                    }
                )

            except Exception as e:

                st.error(f"Error: {e}")

# ============================================================
# NO DOCUMENTS
# ============================================================

elif question and chain is None:

    st.warning("Please upload documents first.")