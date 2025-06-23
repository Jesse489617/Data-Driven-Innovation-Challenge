from transformers import pipeline, AutoTokenizer

MODEL_NAME = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
summarizer = pipeline("summarization", model=MODEL_NAME, tokenizer=tokenizer)

MAX_INPUT_TOKENS = 480 

def truncate_text(text, max_tokens=MAX_INPUT_TOKENS):
    tokens = tokenizer(text, truncation=True, max_length=max_tokens, return_tensors="pt")
    truncated_ids = tokens["input_ids"][0]
    return tokenizer.decode(truncated_ids, skip_special_tokens=True)

def summarize_chunks(paragraphs):
    summaries = []
    for i in range(0, len(paragraphs), 3):
        chunk = " ".join(paragraphs[i:i+3]).strip()
        if len(chunk) < 10:
            continue

        truncated_chunk = truncate_text(chunk, MAX_INPUT_TOKENS)

        input_length = len(tokenizer(truncated_chunk)["input_ids"])
        max_len = min(100, int(input_length * 0.8))
        min_len = min(50, int(input_length * 0.4))

        try:
            summary = summarizer(truncated_chunk, max_length=max_len, min_length=min_len, do_sample=False)[0]['summary_text']
            summaries.append(summary)
        except Exception as e:
            print(f"Skipping chunk due to error: {e}")

    return "\n".join(summaries)
