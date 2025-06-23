
from scraping.final_scraper import scrape_fandom_page
from summarization.final_summarizer import summarize_chunks
from questioning.final_questioner import raw_ask_question 

MAX_INPUT_TOKENS = 480

retriever = None

def truncate_prompt(prompt, tokenizer, max_tokens=MAX_INPUT_TOKENS):
    tokens = tokenizer(prompt, truncation=True, max_length=max_tokens, return_tensors='pt')
    return tokenizer.decode(tokens['input_ids'][0], skip_special_tokens=True)

def process_url(url):
    data = scrape_fandom_page(url)
    chunks = data['chunks'] 

    print("done with chuncks")
    return chunks

def summarization(paragraphs):
    summary = summarize_chunks(paragraphs)
    print("done with summarization")
    return summary

def ask_question(question, retriever):
    return raw_ask_question(question, retriever)
