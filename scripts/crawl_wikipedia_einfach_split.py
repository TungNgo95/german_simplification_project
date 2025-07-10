import requests
from bs4 import BeautifulSoup
import time
import random

urls = [
    "https://de.wikipedia.org/wiki/Wikipedia:Sprachversion_in_einfacher_Sprache",
    "https://de.wikipedia.org/wiki/Albert_Einstein",
    "https://de.wikipedia.org/wiki/Sonne",
    "https://de.wikipedia.org/wiki/Wasser"
]

def crawl_article(url):
    print(f"Crawling: {url}")
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Fehler beim Abrufen der Seite: {url}")
        return None

    soup = BeautifulSoup(response.text, "html.parser")
    title = soup.find("h1", id="firstHeading").get_text().strip()
    paragraphs = soup.find_all("p")
    text = "\n".join(p.get_text().strip() for p in paragraphs if p.get_text().strip())
    
    return (title, text)

articles = []
for url in urls:
    result = crawl_article(url)
    if result:
        articles.append(result)
    time.sleep(1)  # Chờ nhẹ để tránh bị chặn

print(f"Gesamt: {len(articles)} Artikel geladen.")

random.shuffle(articles)
n = len(articles)
n_train = int(n * 0.7)
n_val = int(n * 0.15)

train = articles[:n_train]
val = articles[n_train:n_train + n_val]
test = articles[n_train + n_val:]

splits = {"train.txt": train, "val.txt": val, "test.txt": test}

for filename, data in splits.items():
    with open(filename, "w", encoding="utf-8") as f:
        for title, text in data:
            f.write(f"{title}\t{text}\n\n")
    print(f"{filename} gespeichert mit {len(data)} Artikeln.")

print("Dataset fertig!")