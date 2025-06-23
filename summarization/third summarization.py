from transformers import pipeline
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from typing import List

summarizer = pipeline("summarization", model="google/flan-t5-base", tokenizer="google/flan-t5-base")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def dynamic_summary_length(text, scale=0.5, max_cap=300):
    word_count = len(text.split())
    max_length = min(int(word_count * scale), max_cap)
    min_length = max(int(max_length * 0.5), 20)
    return max_length, min_length

def summarize_chunks(chunks: List[dict], num_clusters: int = 5):
    texts = [chunk["text"] for chunk in chunks if len(chunk["text"].strip()) > 50]
    if not texts:
        return "No content available to summarize."

    embeddings = embedder.encode(texts)
    first_embedding = embeddings[0]

    kmeans = KMeans(n_clusters=min(num_clusters, len(texts)), random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)

    first_label = labels[0]

    target_texts = [t for l, t in zip(labels, texts) if l == first_label]
    combined_text = " ".join(target_texts)[:3000] 

    max_len, min_len = dynamic_summary_length(combined_text, scale=0.6, max_cap=350)
    try:
        summary = summarizer(
            combined_text,
            max_length=max_len,
            min_length=min_len,
            do_sample=False
        )[0]['summary_text']
        return summary
    except Exception as e:
        return f"Summarization failed: {e}"
