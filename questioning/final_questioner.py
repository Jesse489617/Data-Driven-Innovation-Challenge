from transformers import pipeline, AutoTokenizer

qa = pipeline("text2text-generation", model="google/flan-t5-base")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
MAX_INPUT_TOKENS = 480

def truncate_prompt(prompt, tokenizer, max_tokens=MAX_INPUT_TOKENS):
    tokens = tokenizer(prompt, truncation=True, max_length=max_tokens, return_tensors='pt')
    return tokenizer.decode(tokens['input_ids'][0], skip_special_tokens=True)

def raw_ask_question(question, retriever):
    if retriever is None:
        return "No retriever context available."

    hits = retriever.query(question, top_k=3)
    if not hits:
        return "No relevant context found."

    context = "\n".join([h['text'] if isinstance(h, dict) else str(h) for h in hits])
    prompt = f"Answer the question based on the text below:\n\n{context}\n\nQuestion: {question}\nAnswer:"
    prompt = truncate_prompt(prompt, tokenizer)
    input_tokens = tokenizer(prompt, return_tensors='pt')['input_ids'].shape[1]
    max_output_tokens = min(200, int(input_tokens * 0.5) + 20)

    try:
        return qa(prompt, max_length=max_output_tokens, do_sample=False)[0]['generated_text']
    except Exception as e:
        return f"Error: {e}"
