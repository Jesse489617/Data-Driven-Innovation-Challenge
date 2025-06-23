from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class Retriever:
    def __init__(self, chunks):
        self.chunks = chunks
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = self.model.encode([chunk['text'] for chunk in chunks], convert_to_tensor=True)

    # def query(self, q, top_k=3):
    #     query_embedding = self.model.encode(q, convert_to_tensor=True)
    #     scores = cosine_similarity([query_embedding.cpu().numpy()], self.embeddings.cpu().numpy())[0]
    #     top_indices = np.argsort(scores)[::-1][:top_k]
    #     return [self.chunks[i]['text'] for i in top_indices]
    
    def query(self, q, top_k=3):
        query_embedding = self.model.encode(q, convert_to_tensor=True)
        scores = cosine_similarity([query_embedding.cpu().numpy()], self.embeddings.cpu().numpy())[0]
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [self.chunks[i] for i in top_indices]
