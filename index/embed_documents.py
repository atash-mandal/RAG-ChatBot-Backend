from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from utils import split_documents
from langchain_core.documents import Document
import os

# Create embedding model
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

documents = []

# Load crawled text
with open("./data/crawled_pages/crawled.txt", encoding="utf-8") as f:
    text = f.read()
    documents.append(Document(page_content=text))

# Load parsed PDF text
with open("./data/parsed_pdfs.txt", encoding="utf-8") as f:
    text = f.read()
    documents.append(Document(page_content=text))

print("Loaded documents...")
chunks = split_documents(documents)
print(f"Split into {len(chunks)} chunks.")

# Generate embeddings and build the FAISS index incrementally
batch_size = 500  # Adjust based on memory
batched_chunks = [chunks[i:i + batch_size] for i in range(0, len(chunks), batch_size)]

# Initialize FAISS index with the first batch
first_batch = batched_chunks[0]
print(f"Initializing FAISS index with the first batch of {len(first_batch)} chunks")
index = FAISS.from_documents(first_batch, embedding)

# Process and add embeddings in batches
for i, batch in enumerate(batched_chunks[1:], start=1):
    print(f"Processing batch {i + 1}/{len(batched_chunks)} with {len(batch)} chunks")
    
    # Generate embeddings for the current batch
    batch_vectorstore = FAISS.from_documents(batch, embedding)
    
    # Instead of using add_vectors, merge the two vector stores
    index.merge_from(batch_vectorstore)

# Save the final vector store
index.save_local("index/faiss_index")
print("Index saved.")
