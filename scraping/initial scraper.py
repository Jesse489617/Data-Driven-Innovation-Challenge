from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup

def scrape_fandom_page(url):
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(url)
        page.wait_for_load_state("networkidle")

        html = page.content()
        browser.close()

    soup = BeautifulSoup(html, 'html.parser')
    title = soup.find('h1').text.strip()

    content_div = soup.find('div', {'class': 'mw-parser-output'})
    paragraphs = []
    for tag in content_div.find_all(['p', 'h2', 'h3']):
        text = tag.get_text(strip=True)
        if len(text) > 0:
            paragraphs.append(text)

    return {
        'title': title,
        'url': url,
        'content': paragraphs
    }




