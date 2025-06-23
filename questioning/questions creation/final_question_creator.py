from transformers import pipeline, AutoTokenizer
from typing import List, Dict
import csv
import os
import re
import pandas as pd

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

def fix_spacing(text: str) -> str:
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    text = re.sub(r'\)([A-Za-z])', r') \1', text)
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()

def clean_section(section: str) -> str:
    if not section:
        return "Unknown"
    return section.replace("[]", "").strip().capitalize()

def clean_context(text: str) -> str:
    text = re.sub(r"\(\s*,?.*?\)", "", text)
    text = re.sub(r"née\s*Hyga", "née Hyūga", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()

def extract_questions_from_output(output: str) -> List[str]:
    questions = []
    for line in output.strip().splitlines():
        if line.strip().lower().startswith("q:"):
            questions.append(line.strip()[2:].strip())
    return questions

def generate_questions(chunk: Dict, num_questions: int = 3, simple_prompt: bool = False) -> List[str]:
    text_fixed = fix_spacing(chunk['text'])
    truncated_text = truncate_text(text_fixed, max_tokens=MAX_INPUT_TOKENS - 100)

    if simple_prompt:
        prompt = (
            f"Generate {num_questions} simple questions from the text below, each starting with 'Q:' on a new line.\n\n"
            f"TEXT:\n{truncated_text}"
        )
    else:
        prompt = (
            f"You are an expert in anime trivia. Given the text below, generate exactly {num_questions} "
            f"questions only, each starting with 'Q:' on a new line.\n\nTEXT:\n{truncated_text}"
        )

    print("\n Question prompt sent to model:\n", prompt[:500], "..." if len(prompt) > 500 else "")
    try:
        result = qa_generator(prompt)
        output = result[0]['generated_text']
        print("\nRaw questions output:\n", output)
    except Exception as e:
        print(f"Failed to generate questions: {e}")
        return []

    questions = extract_questions_from_output(output)
    print(f"Extracted {len(questions)} questions.")
    return questions

def generate_answer(chunk: Dict, question: str) -> str:
    base_prompt = (
        f"You are an expert in anime trivia. Given the text below and a question, provide a clear and concise answer. "
        f"Format your output as 'A: ...'\n\nTEXT:\n"
    )
    text_fixed = fix_spacing(chunk['text'])
    truncated_text = truncate_text(text_fixed, max_tokens=MAX_INPUT_TOKENS - 150)
    prompt = f"{base_prompt}{truncated_text}\n\nQ: {question}\nA:"

    print("\n Answer prompt sent to model:\n", prompt[:500], "..." if len(prompt) > 500 else "")
    try:
        result = qa_generator(prompt)
        output = result[0]['generated_text']
        print("\nRaw answer output:\n", output)
    except Exception as e:
        print(f"Failed to generate answer: {e}")
        return ""

    answer_match = re.search(r"A:\s*(.*)", output, re.IGNORECASE | re.DOTALL)
    answer = answer_match.group(1).strip() if answer_match else output.strip()
    return answer

def save_qa_pairs_to_csv(qa_pairs: List[Dict], file_path="/data/qa_dataset.csv"):
    file_exists = os.path.isfile(file_path)
    with open(file_path, mode="a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["question", "answer", "section", "context"])
        if not file_exists:
            writer.writeheader()
        for row in qa_pairs:
            writer.writerow(row)

def save_qa_pairs_to_excel(qa_pairs: List[Dict], file_path="/data/qa_dataset.xlsx"):
    df = pd.DataFrame(qa_pairs)
    if os.path.exists(file_path):
        existing_df = pd.read_excel(file_path)
        combined_df = pd.concat([existing_df, df], ignore_index=True)
        combined_df.to_excel(file_path, index=False)
    else:
        df.to_excel(file_path, index=False)
