import json
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer

from chunking import create_chunks


MODEL_NAME = "all-MiniLM-L6-v2"

INDEX_DIR = Path("part3_support_agent/rag/index")
INDEX_FILE = INDEX_DIR / "policy.index"
CHUNKS_FILE = INDEX_DIR / "chunks.json"


def create_embeddings():
    chunks = create_chunks()

    texts = [chunk["text"] for chunk in chunks]

    model = SentenceTransformer(MODEL_NAME)

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    return chunks, embeddings


def save_faiss_index(chunks, embeddings):
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    faiss.write_index(index, str(INDEX_FILE))

    with open(CHUNKS_FILE, "w", encoding="utf-8") as file:
        json.dump(chunks, file, indent=2, ensure_ascii=False)

    return index


if __name__ == "__main__":
    chunks, embeddings = create_embeddings()

    index = save_faiss_index(chunks, embeddings)

    print("Embedding model:", MODEL_NAME)
    print("Chunks:", len(chunks))
    print("Embedding shape:", embeddings.shape)
    print("FAISS vectors:", index.ntotal)
    print("Index saved:", INDEX_FILE)
    print("Chunk metadata saved:", CHUNKS_FILE)