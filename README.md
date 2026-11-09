# 📄 RAG-DOCUMENT-Chatbot

An intelligent **Retrieval-Augmented Generation (RAG)** chatbot built with **Streamlit, LangChain, ChromaDB, and SentenceTransformers**. The application allows users to upload multiple PDF documents and ask questions in natural language. The chatbot retrieves relevant information from the uploaded PDFs and generates accurate answers with source citations.

---

## ✨ Features

* 📂 Upload **multiple PDF files**
* 🔍 Ask questions in natural language
* 📑 **Source citations** with PDF name and page number
* 🎛️ Adjustable **Chunk Size**, **Chunk Overlap**, and **Top-K retrieval**
* 🧠 **Conversational memory** for follow-up questions
* 🔗 **Hybrid Retrieval** (Vector Search + BM25 Keyword Search)
* 🎯 **Reranking** using SentenceTransformer (`all-MiniLM-L6-v2`)
* 📊 Automatic evaluation metrics display
* ⚡ Fast interactive UI with Streamlit

---

## 🛠️ Tech Stack

* **Python**
* **Streamlit**
* **LangChain**
* **ChromaDB**
* **SentenceTransformers**
* **BM25 (rank-bm25)**
* **PyPDFLoader**

---

## 📁 Project Structure

```text
Rag_project/
│
├── app.py
├── app_level1.py
├── services/
│   ├── rag.py
│   ├── reranker.py
│   ├── embeddings.py
│   ├── llm.py
│   └── chunker.py
├── loaders/
│   └── pdf_loader.py
├── prompts/
│   └── prompt.py
├── data/
└── README.md
```

---

## 🚀 Installation

```bash
git clone https://github.com/varshu234/Rag_project.git
cd Rag_project
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it:

**Windows**

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

Open the local URL shown in the terminal (usually `http://localhost:8501`).

---

## 📚 Levels Completed

### ✅ Level 1

* Changed embedding models and reasoning LLMs
* Compared response speed across models
* Experimented with chunking strategy, chunk size, and overlap

### ✅ Level 2

* Multiple PDF upload
* Upload PDFs through Streamlit
* Source citations with PDF and page number
* Top-K retrieval slider

### ✅ Level 3

* Conversational memory
* Hybrid retrieval (Vector + BM25)
* Reranking pipeline (Top 10 → Top 3 → LLM)
* Automatic evaluation metrics display

---

## 🔄 Retrieval Pipeline

```text
User Question
      ↓
Hybrid Retriever
(Vector + BM25)
      ↓
Top 10 Chunks
      ↓
Reranker
(SentenceTransformer)
      ↓
Top 3 Chunks
      ↓
LLM
      ↓
Final Answer + Sources
```

---

## 📊 Evaluation Metrics

The chatbot automatically displays running metrics after each question:

* Retrieval Accuracy
* Precision
* Recall
* Answer Correctness

---

## 🖼️ Example Output

```text
Q: What is Siva's CGPA?

Answer: 8.0 / 10.0

Sources:
- siva__resume.pdf — Page 1
```

---

## ⭐ Acknowledgement

Developed as part of a **RAG / Generative AI project** to demonstrate document understanding, retrieval optimization, and conversational question answering using modern AI tools.
## 🚀 Level 4 – Image RAG

Extended the RAG pipeline to understand **visual content inside PDFs**.

### ✨ Features

* Extracts images, tables, charts, and diagrams from PDF pages
* Adds image-related information to the vector database
* Retrieves visual context along with text context
* Answers questions about diagrams, charts, and tables
* Provides source citations with page numbers

### 🛠️ Technologies Used

* PyMuPDF (`fitz`)
* LangChain
* ChromaDB
* Streamlit

### 💬 Example Questions

* *What diagram is shown in the PDF?*
* *Describe the chart on page 3.*
* *What information is present in the table?*

---

## 🚀 Level 5 – Multi-Document RAG

Enhanced the chatbot to support **multiple document formats** in a single application.

### 📂 Supported File Types

* PDF
* TXT
* DOCX
* CSV
* PPTX

### ✨ Features

* Upload multiple document types simultaneously
* Unified document loading pipeline
* Text extraction from Word, PowerPoint, CSV, and text files
* Search across all uploaded documents
* Source-aware answers with file names
* Works together with Level 4 image retrieval

### 🛠️ Technologies Used

* `python-docx`
* `pandas`
* `python-pptx`
* LangChain
* ChromaDB
* Streamlit

### 💬 Example Questions

* *Summarize the DOCX report.*
* *What values are present in the CSV file?*
* *What topics are discussed in the PPTX presentation?*

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

Upload documents and start asking questions through the Streamlit interface.

---

## 📌 Final Outcome

The project now supports:

* ✅ Text-based RAG
* ✅ Multiple PDF upload
* ✅ Source citations
* ✅ Configurable Top-K retrieval
* ✅ Conversational memory
* ✅ Hybrid retrieval
* ✅ Reranking
* ✅ Retrieval evaluation metrics
* ✅ Image-aware PDF understanding
* ✅ Multi-document format support (PDF, TXT, DOCX, CSV, PPTX)

This completes **Levels 1–5** of the RAG enhancement project.

