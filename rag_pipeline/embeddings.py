from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL

_embedding_model = None

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    return _embedding_model   


def get_embeddings(text):
    model=get_embedding_model()
    embedding = model.encode(text)
    return embedding