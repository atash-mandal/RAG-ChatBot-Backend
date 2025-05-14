# 🧠 RAG Chatbot Backend

This project is the backend for a Retrieval-Augmented Generation (RAG) Chatbot trained on customer support content from [Angel One](https://www.angelone.in/support) and insurance PDFs. It uses FastAPI, LangChain, HuggingFace embeddings, and FAISS to return intelligent, context-aware responses.

---

## 🚀 Features

- Web page crawling and PDF parsing
- Embedding with HuggingFace sentence transformers
- Vector search using FAISS
- FastAPI endpoint for querying the chatbot
- Easy deployment on Render

---

## 🛠 Tech Stack

- **Python 3.11+**
- **FastAPI** – API framework
- **LangChain** – RAG orchestration
- **FAISS** – Vector search
- **HuggingFace Sentence Transformers** – Text embedding
- **BeautifulSoup4** – HTML parsing
- **PyMuPDF / PDFMiner** – PDF parsing
- **Uvicorn** – ASGI server

---

## 📂 Project Structure

.
├── data/
│ ├── crawled_pages/ # Web page HTML/text
│ ├── pdfs/ # Input insurance PDFs
│ └── parsed/ # Output parsed PDF text
├── faiss_index/ # FAISS vector store (generated)
├── ingest/
│ ├── crawl_web.py # Recursively crawl support pages
│ └── parse_pdfs.py # Extract text from PDFs
├── index/
│ ├── embed_documents.py # Embed and index content
│ └── utils.py # Helpers
├── main.py # FastAPI server
├── requirements.txt # All Python dependencies
└── README.md # You're here

## ⚙️ Setup Instructions

### 1. Clone the Repo

```bash
git clone https://github.com/your-org/rag-chatbot-backend.git
cd rag-chatbot-backend
```

### 2. Setup virtual env

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
```

### 3. Update pip and install requirements

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Crawl Webpages

```bash
python ingest/crawl_web.py
```

### 5. Parse Pdfs

```bash
python ingest/parse_pdfs.py
```

### 6. Embed the documents

```bash
python index/embed_documents.py
```

## Example API Request

```bash
curl -X POST http://localhost:10000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "How do I reset my password?"}'
```