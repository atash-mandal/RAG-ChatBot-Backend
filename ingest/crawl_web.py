import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import os
import time

BASE_URL = "https://www.angelone.in/support"
MAX_DEPTH = 10
visited = set()
documents = []

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; CustomCrawler/1.0; +https://example.com/bot)"
}

def is_valid_url(url):
    parsed = urlparse(url)
    return (
        parsed.scheme in ("http", "https")
        and BASE_URL in url
        and not parsed.fragment  # Skip URLs with # fragments
    )

def clean_html(html):
    soup = BeautifulSoup(html, "html.parser")

    # Remove unwanted tags
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    # Remove layout/navigation elements
    for tag in soup.find_all(True, class_=["footer", "header", "navbar", "nav", "menu"]):
        tag.decompose()

    # Extract visible clean text
    text = soup.get_text(separator=" ", strip=True)
    return " ".join(text.split())

def crawl(url, depth):
    if depth > MAX_DEPTH or url in visited:
        return

    visited.add(url)

    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        content_type = response.headers.get("Content-Type", "")

        if "text/html" not in content_type:
            return

        print(f"Crawling: {url} (depth={depth})")
        clean_text = clean_html(response.text)
        if clean_text:
            documents.append(clean_text)

        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup.find_all("a", href=True):
            next_url = urljoin(url, tag["href"].split("?")[0])
            if is_valid_url(next_url) and next_url not in visited:
                crawl(next_url, depth + 1)

        time.sleep(0.5)  # Be polite
    except Exception as e:
        print(f"Error crawling {url}: {e}")

if __name__ == "__main__":
    print("Crawling Web Pages...")
    crawl(BASE_URL, 0)

    output_dir = "./data/crawled_pages"
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "crawled.txt"), "w", encoding="utf-8") as f:
        for doc in documents:
            f.write(doc + "\n")

    print("Crawling Completed.")
    print(f"Total pages crawled: {len(documents)}")
