from pathlib import Path
import json
import os
import sys

import faiss
import numpy as np

if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from agent.data import policies as POLICIES_PATH
from config import INDEX_PATH, DOCS_PATH
from rag_pipeline.embeddings import get_embeddings

DATA_DIR = Path(__file__).resolve().parent / "data"

def build_vector_store():
    all_chunks = []
    metadata = []

    with open(DATA_DIR / "products.txt", "r", encoding="utf-8") as f:
        text = f.read()
        chunks = text.split('\n\n')

        for chunk in chunks:
            all_chunks.append(chunk)
            metadata.append({'domain': 'product'})

    with open(POLICIES_PATH, "r", encoding="utf-8") as f:
        policies = json.load(f)

        for policy_data in policies.values():
            all_chunks.append(f"{policy_data['title']}\n\n{policy_data['content']}")
            metadata.append({'domain': "policy"})


    embeddings = get_embeddings(all_chunks)
    embeddings = np.array(embeddings).astype("float32")

    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    faiss.write_index(index,INDEX_PATH)

    np.save(DOCS_PATH,np.array(list(zip(all_chunks,metadata)),dtype=object))

    print("✅ Vector store rebuilt successfully.")

    return index , list(zip(all_chunks,metadata))

index = None
documents = None
def load_vector_store():
    global index,documents

    if index is not None and documents is not None:
        return index, documents

    if not os.path.exists(INDEX_PATH) or not os.path.exists(DOCS_PATH):
        return None,None

    try:
        index = faiss.read_index(INDEX_PATH)
        documents= np.load(DOCS_PATH,allow_pickle=True).tolist()
        return index, documents
    except Exception as e:
        print(f'[VECTOR LOAD ERROR] {e}')
        return None,None

def delete_vector_store():
    global index, documents
    index = None
    documents = None
    if os.path.exists(INDEX_PATH):
        os.remove(INDEX_PATH)
        print("🗑 Deleted existing FAISS index.")

    if os.path.exists(DOCS_PATH):
        os.remove(DOCS_PATH)
        print("🗑 Deleted existing document store.")

def rebuid_vector_store():
    choice = input('"Rebuild vector store? (yes/no): ').strip().lower()

    if choice == 'yes':
        delete_vector_store()
        build_vector_store()
    else:
        print("🚫 Rebuild cancelled. Exiting.")

if __name__ == '__main__':
    rebuid_vector_store()