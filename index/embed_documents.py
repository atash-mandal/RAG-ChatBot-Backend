import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from utils import split_documents

# Configure environment
os.environ["HF_HOME"] = os.path.abspath("D:/hf_cache")

def load_documents():
    """Load and return documents from text files."""
    files = [
        "./data/crawled_pages/crawled.txt",
        "./data/parsed_pdfs.txt"
    ]
    
    documents = []
    for file_path in files:
        with open(file_path, encoding="utf-8") as f:
            documents.append(Document(page_content=f.read()))
    
    return documents

def build_faiss_index():
    """Build and save FAISS index from documents."""
    # Initialize embedding model
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Load and split documents
    print("Loading documents...")
    documents = load_documents()
    chunks = split_documents(documents)
    print(f"Split into {len(chunks)} chunks")
    
    # Build index in batches
    batch_size = 500
    print(f"Building FAISS index with batch size {batch_size}")
    
    # Initialize with first batch
    index = FAISS.from_documents(chunks[:batch_size], embedding)
    
    # Add remaining chunks in batches
    for i in range(batch_size, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        print(f"Processing batch {i//batch_size + 1}, chunks {i}-{i + len(batch)}")
        
        batch_index = FAISS.from_documents(batch, embedding)
        index.merge_from(batch_index)
    
    # Save index
    index.save_local("index/faiss_index")
    print("Index saved successfully")
    
    return index

if __name__ == "__main__":
    build_faiss_index()