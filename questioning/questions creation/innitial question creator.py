from transformers import pipeline, AutoTokenizer
from typing import List, Dict
import csv
import os
import re

qa_generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    max_length=512,
    do_sample=True,
    temperature=0.7
)
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")

MAX_INPUT_TOKENS = 480

def truncate_text(text, max_tokens=MAX_INPUT_TOKENS):
    tokens = tokenizer(text, truncation=True, max_length=max_tokens, return_tensors='pt')
    return tokenizer.decode(tokens['input_ids'][0], skip_special_tokens=True)

def clean_section(section: str) -> str:
    if not section:
        return "Unknown"
    return section.replace("[]", "").strip().capitalize()

def clean_context(text: str) -> str:
    text = re.sub(r"\(\s*,?.*?\)", "", text)
    text = re.sub(r"née\s*Hyga", "née Hyūga", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()

def extract_qa_pairs_from_output(output: str) -> List[Dict]:
    qa_pairs = []
    current_q = None

    lines = output.strip().splitlines()
    for line in lines:
        if line.strip().lower().startswith("q:"):
            current_q = line.strip()[2:].strip()
        elif line.strip().lower().startswith("a:") and current_q:
            current_a = line.strip()[2:].strip()
            if current_q and current_a:
                qa_pairs.append((current_q, current_a))
                current_q = None
    return qa_pairs

def generate_qa_pairs(chunk: Dict, num_questions: int = 3) -> List[Dict]:
    base_prompt = (
        f"You are an expert in anime trivia. Given the text below, generate exactly {num_questions} "
        f"question-answer pairs in the format:\nQ: ...\nA: ...\n\nTEXT:\n"
    )

    truncated_text = truncate_text(chunk['text'], max_tokens=MAX_INPUT_TOKENS - 100)
    full_prompt = f"{base_prompt}{truncated_text}"

    print("\n Prompt sent to model:\n", full_prompt[:500], "..." if len(full_prompt) > 500 else "")

    try:
        result = qa_generator(full_prompt)
        output = result[0]['generated_text']
        print("\nRaw model output:\n", output)
    except Exception as e:
        print(f"Failed to generate QA: {e}")
        return []

    raw_pairs = extract_qa_pairs_from_output(output)
    cleaned_context = clean_context(truncated_text)
    cleaned_section = clean_section(chunk.get('section', ''))

    unique_set = set()
    final_pairs = []
    for q, a in raw_pairs:
        key = (q.strip().lower(), cleaned_context)
        if key not in unique_set:
            final_pairs.append({
                "question": q.strip(),
                "answer": a.strip(),
                "section": cleaned_section,
                "context": cleaned_context
            })
            unique_set.add(key)

    print(f"Extracted {len(final_pairs)} QA pairs.")
    return final_pairs

def save_qa_pairs_to_csv(qa_pairs: List[Dict], file_path="/data/qa_dataset.csv"):
    file_exists = os.path.isfile(file_path)
    with open(file_path, mode="a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["question", "answer", "section", "context"])
        if not file_exists:
            writer.writeheader()
        for row in qa_pairs:
            writer.writerow(row)
