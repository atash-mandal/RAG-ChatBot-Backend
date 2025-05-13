import os
from langchain.document_loaders import PyPDFLoader

pdf_dir = "./data/pdfs"
documents = []

for file in os.listdir(pdf_dir):
    if file.endswith(".pdf"):
        loader = PyPDFLoader(os.path.join(pdf_dir, file))
        documents.extend(loader.load())

with open("./data/parsed_pdfs.txt", "w", encoding="utf-8") as f:
    for doc in documents:
        f.write(doc.page_content + "\n")