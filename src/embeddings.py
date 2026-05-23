import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings
from typing import List


class SentenceTransformerEmbeddings(Embeddings):

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        print(f"Loaded SentenceTransformer: {model_name}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts, show_progress_bar=True)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        embedding = self.model.encode([text])
        return embedding[0].tolist()


def get_embedding_model():
    return SentenceTransformerEmbeddings()


if __name__ == "__main__":
    model = get_embedding_model()

