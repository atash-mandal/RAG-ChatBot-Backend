import os
from langchain_community.document_loaders import PyPDFLoader

pdf_dir = "./data/pdfs"
output_file = "./data/parsed_pdfs.txt"
documents = []

print("Parsing PDF Files...")

if not os.path.exists(pdf_dir):
    print(f"Directory not found: {pdf_dir}")
    exit(1)

for filename in os.listdir(pdf_dir):
    file_path = os.path.join(pdf_dir, filename)

    if os.path.isfile(file_path) and filename.lower().endswith(".pdf"):
        try:
            print(f"Loading: {filename}")
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            documents.extend(docs)
        except Exception as e:
            print(f"Failed to parse {filename}: {e}")

# Ensure output directory exists
os.makedirs(os.path.dirname(output_file), exist_ok=True)

# Save all extracted text
with open(output_file, "w", encoding="utf-8") as f:
    for doc in documents:
        f.write(doc.page_content.strip() + "\n\n")

print(f"Parsing Completed. {len(documents)} documents written to {output_file}")
