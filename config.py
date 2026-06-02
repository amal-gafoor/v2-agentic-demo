import os
from pathlib import Path

from dotenv import load_dotenv 
from groq import Groq
from sentence_transformers import CrossEncoder

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
GROQ_MODEL = 'llama-3.3-70b-versatile'
RERANKER_MODEL = 'cross-encoder/ms-marco-MiniLM-L-6-v2'

TOP_K = 3
MAX_HISTORY = 6

INDEX_PATH = "rag_pipeline/data/faiss_index.index"
DOCS_PATH = "rag_pipeline/data/documents.npy"

BASE_DIR = Path(__file__).resolve().parent
INDEX_PATH = str(BASE_DIR / "rag_pipeline" / "data" / "faiss_index.index")
DOCS_PATH = str(BASE_DIR / "rag_pipeline" / "data" / "documents.npy")


SESSION_DIR = str(BASE_DIR / "sessions")
LOG_DIR = str(BASE_DIR / "logs")


_groq_client = None

def get_groq_client():

    global _groq_client

    if _groq_client is None:
        _groq_client = Groq(api_key=GROQ_API_KEY)
        
    return _groq_client    


_reranker_model = None

def get_reranker_model():
    global _reranker_model

    if _reranker_model is None:
        _reranker_model = CrossEncoder(RERANKER_MODEL)
    
    return _reranker_model
