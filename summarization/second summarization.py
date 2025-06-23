from sentence_transformers import SentenceTransformer
from transformers import pipeline
from sklearn.cluster import KMeans

embedder = SentenceTransformer("all-MiniLM-L6-v2")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_chunks(chunks, num_clusters=5):
    """
    Group semantically similar chunks using embeddings, then summarize each group.
    :param chunks: List of chunk dicts with 'text' field
    :param num_clusters: Number of summary clusters (can be dynamic)
    """
    texts = [chunk['text'] for chunk in chunks if len(chunk['text']) > 50]

    if len(texts) == 0:
        return "Not enough content to summarize."

    print(f"Embedding {len(texts)} chunks...")
    embeddings = embedder.encode(texts)

    num_clusters = min(num_clusters, len(texts))
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings)

    clustered_texts = {i: [] for i in range(num_clusters)}
    for idx, label in enumerate(cluster_labels):
        clustered_texts[label].append(texts[idx])

    summaries = []
    for cluster_id, group in clustered_texts.items():
        combined_text = " ".join(group)
        if len(combined_text) > 1000:
            combined_text = combined_text[:1000] 
        try:
            summary = summarizer(combined_text, max_length=100, min_length=30, do_sample=False)[0]['summary_text']
        except Exception as e:
            summary = f"Summarization failed for cluster {cluster_id}: {e}"
        summaries.append(f"Summary {cluster_id+1}:\n{summary}")

    return "\n\n".join(summaries)
