import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import os
import time

BASE_URL = "https://www.angelone.in/support"
MAX_DEPTH = 10
visited = set()
documents = []

def is_valid_url(url):
    parsed = urlparse(url)
    return parsed.scheme in ("http", "https") and BASE_URL in url

def clean_html(html):
    soup = BeautifulSoup(html, "html.parser")

    # Remove script, style, and unwanted tags
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    # Remove elements that are likely to be layout or navigation
    for tag in soup.find_all(True, {"class": ["footer", "header", "navbar", "nav", "menu"]}):
        tag.decompose()

    # Extract visible text
    text = soup.get_text(separator=" ", strip=True)

    # Remove multiple spaces and weird characters
    return " ".join(text.split())

def crawl(url, depth):
    if depth > MAX_DEPTH or url in visited:
        return
    try:
        response = requests.get(url, timeout=10)
        visited.add(url)

        if "text/html" not in response.headers.get("Content-Type", ""):
            return

        clean_text = clean_html(response.text)
        if clean_text:
            documents.append(clean_text)

        soup = BeautifulSoup(response.text, "html.parser")
        for link in soup.find_all("a", href=True):
            next_url = urljoin(url, link["href"])
            if is_valid_url(next_url):
                crawl(next_url, depth + 1)

        time.sleep(0.5)  # Be polite
    except Exception as e:
        print(f"Error crawling {url}: {e}")

print("Crawling Web Pages...")
crawl(BASE_URL, 0)

os.makedirs("./data/crawled_pages", exist_ok=True)
with open("./data/crawled_pages/crawled.txt", "w", encoding="utf-8") as f:
    for doc in documents:
        f.write(doc + "\n")

print("Crawling Completed.")
print(f"Total pages crawled: {len(documents)}")
