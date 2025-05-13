from langchain.document_loaders import RecursiveUrlLoader

loader = RecursiveUrlLoader(
    url="https://www.angelone.in/support",
    max_depth=10,
    extractor=lambda x: x  # HTML is already a string
)

documents = loader.load()

with open("./data/crawled_pages/crawled.txt", "w", encoding="utf-8") as f:
    for doc in documents:
        f.write(doc.page_content + "\n")
