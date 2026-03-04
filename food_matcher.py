import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class FoodMatcher:
    def __init__(self, db_path):
        # Embedding-Modell von Hugging Face
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        with open(db_path, "r") as f:
            self.food_db = json.load(f)

        self.food_names = list(self.food_db.keys())
        self.food_embeddings = self.model.encode(self.food_names)

    def match(self, query):
        # query in Embedding umwandeln
        query_embedding = self.model.encode([query])
        similarities = cosine_similarity(query_embedding, self.food_embeddings)[0]
        best_index = np.argmax(similarities)
        return self.food_names[best_index]