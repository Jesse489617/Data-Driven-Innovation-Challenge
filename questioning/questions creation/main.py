import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

from scraping.final_scraper import scrape_fandom_page
from summarization.final_summarizer import summarize_chunks
from questioning.final_questioner import generate_questions, generate_answer, save_qa_pairs_to_csv, save_qa_pairs_to_excel, clean_section, clean_context

CSV_FILE = "/data/qa_dataset.csv"
XLSX_FILE = "/data/qa_dataset.xlsx"

def run_pipeline(url):
    print(f"\n Scraping data from: {url}")
    data = scrape_fandom_page(url)
    chunks = data['chunks']
    print(f" Loaded {len(chunks)} chunks from: {data['title']}")

    print("\n Summarizing chunks...")
    summary = summarize_chunks(chunks)
    print("\n Summary:\n", summary)

    print("\n Generating QA pairs...")

    for i, chunk in enumerate(chunks):
        print(f"\n Processing chunk {i + 1}/{len(chunks)}")
        print("📏 Chunk preview:\n", chunk['text'][:300], "..." if len(chunk['text']) > 300 else "")

        questions = generate_questions(chunk, num_questions=3)

        if not questions:
            print(" No questions generated for this chunk. Trying a fallback prompt...")
            questions = generate_questions(chunk, num_questions=1, simple_prompt=True)

            if not questions:
                print(" Still no questions generated after fallback. Skipping this chunk.")
                continue
            else:
                print(f" Fallback generated {len(questions)} question(s).")

        qa_pairs = []
        for q in questions:
            a = generate_answer(chunk, q)
            if not a:
                print(f" No answer generated for question: {q}")
                continue
            qa_pairs.append({
                "question": q,
                "answer": a,
                "section": clean_section(chunk.get("section", "")),
                "context": clean_context(chunk["text"])
            })

        if qa_pairs:
            save_qa_pairs_to_csv(qa_pairs, file_path=CSV_FILE)
            save_qa_pairs_to_excel(qa_pairs, file_path=XLSX_FILE)
            print(f" Saved {len(qa_pairs)} QA pairs.")
        else:
            print(" No valid QA pairs to save for this chunk.")

    print(f"\n Done! QA pairs saved to {CSV_FILE} and {XLSX_FILE}")

if __name__ == "__main__":
    run_pipeline("https://naruto.fandom.com/wiki/Hinata_Hyūga")
