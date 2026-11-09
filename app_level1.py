import os
import time
import streamlit as st

from services.chunker import Chunker
from services.vector_db import VectorDB
from services.rag import RAGService
from loaders.pdf_loader import PDFLoader

# ---------------------------------------------------
# Page Configuration
# ---------------------------------------------------

st.set_page_config(
    page_title="PDF RAG Chatbot",
    page_icon="📄",
    layout="wide"
)

# ---------------------------------------------------
# Sidebar
# ---------------------------------------------------

with st.sidebar:

    st.title("📄 PDF RAG")

    st.markdown("---")

    st.write("### Chunking Settings")

    # Chunking sliders
    chunk_size = st.slider(
        "Chunk Size",
        min_value=200,
        max_value=1500,
        value=500,
        step=100
    )

    chunk_overlap = st.slider(
        "Chunk Overlap",
        min_value=0,
        max_value=500,
        value=100,
        step=50
    )

    st.markdown("---")

    if st.button("🗑️ Clear Chat", key="clear_chat_btn"):

        st.session_state.messages = []

        st.rerun()

    st.markdown("---")

    st.success("Resume Loaded")

# ---------------------------------------------------
# Session State
# ---------------------------------------------------

if "messages" not in st.session_state:

    st.session_state.messages = []

# ---------------------------------------------------
# Load Database
# ---------------------------------------------------

@st.cache_resource
def load_chain(chunk_size, chunk_overlap):

    app_dir = os.path.dirname(__file__)

    # Resume PDF path
    pdf_path = os.path.abspath(
        os.path.join(
            app_dir,
            "data",
            "siva__resume.pdf"
        )
    )

    if not os.path.exists(pdf_path):

        raise FileNotFoundError(
            f"PDF file not found: {pdf_path}"
        )

    # Load PDF
    docs = PDFLoader.load_pdf(pdf_path)

    if not docs:

        raise ValueError(
            "The PDF file could not be read."
        )

    # Apply chunking
    docs = Chunker.split(
        docs,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    # Create unique DB folder for each experiment
    experiment_db = f"chroma_db_{chunk_size}_{chunk_overlap}"

    db_path = os.path.abspath(
        os.path.join(app_dir, experiment_db)
    )

    # Create DB only if it does not exist
    if not os.path.exists(db_path):

        from langchain_chroma import Chroma
        from services.embeddings import EmbeddingModel

        embeddings = EmbeddingModel.load_embeddings()

        Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=db_path
        )

    # Load DB
    from langchain_chroma import Chroma
    from services.embeddings import EmbeddingModel

    db = Chroma(
        persist_directory=db_path,
        embedding_function=EmbeddingModel.load_embeddings()
    )

    # Create RAG chain
    chain = RAGService.create_chain(db)

    return chain

# Load chain with selected chunk settings
chain = load_chain(chunk_size, chunk_overlap)

# ---------------------------------------------------
# Main Page
# ---------------------------------------------------

st.title("📄 PDF Question Answering")

st.caption("Ask questions about your resume")

st.info(
    f"Chunk Size: {chunk_size} | Chunk Overlap: {chunk_overlap}"
)

# ---------------------------------------------------
# Display Chat History
# ---------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

# ---------------------------------------------------
# User Input
# ---------------------------------------------------

question = st.chat_input(
    "Ask a question about your resume..."
)

if question:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):

        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner("Searching resume..."):

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

                    answer = (
                        "I couldn't find that information in the resume."
                    )

                st.markdown(answer)

                st.info(
                    f"⏱ Response time: {latency:.2f} seconds"
                )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer
                    }
                )

            except Exception as e:

                st.error(f"Error: {e}")

                print(e)
