import requests
from bs4 import BeautifulSoup

def scrape_fandom_page(url):
    response = requests.get(url)
    if not response.ok:
        raise Exception(f"Failed to fetch page: {url}")

    soup = BeautifulSoup(response.text, 'html.parser')
    title = soup.find('h1').text.strip()

    content_div = soup.find('div', {'class': 'mw-parser-output'})
    if not content_div:
        raise Exception("Could not find the content container.")

    chunks = []
    current_chunk = []
    current_section_title = "Introduction"

    for tag in content_div.find_all(['h2', 'h3', 'p']):
        if tag.name in ['h2', 'h3']:
            if current_chunk:
                chunks.append({
                    "section": current_section_title,
                    "text": " ".join(current_chunk)
                })
                current_chunk = []
            current_section_title = tag.get_text(strip=True)
        elif tag.name == 'p':
            text = tag.get_text(strip=True)
            if text:
                current_chunk.append(text)

    if current_chunk:
        chunks.append({
            "section": current_section_title,
            "text": " ".join(current_chunk)
        })

    return {
        'title': title,
        'url': url,
        'chunks': chunks  
    }
