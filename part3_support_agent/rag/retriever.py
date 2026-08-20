import json
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"

INDEX_FILE = Path("part3_support_agent/rag/index/policy.index")
CHUNKS_FILE = Path("part3_support_agent/rag/index/chunks.json")


def load_retriever():
    model = SentenceTransformer(MODEL_NAME)

    index = faiss.read_index(str(INDEX_FILE))

    with open(CHUNKS_FILE, "r", encoding="utf-8") as file:
        chunks = json.load(file)

    return model, index, chunks


def retrieve(query, top_k=3):
    model, index, chunks = load_retriever()

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    scores, indices = index.search(query_embedding, top_k)

    results = []

    for score, index_position in zip(scores[0], indices[0]):
        if index_position == -1:
            continue

        chunk = chunks[index_position]

        results.append(
            {
                "chunk_id": chunk["chunk_id"],
                "document_id": chunk["document_id"],
                "title": chunk["title"],
                "text": chunk["text"],
                "score": float(score),
            }
        )

    return results


if __name__ == "__main__":
    query = "What is the return window for apparel products?"

    results = retrieve(query)

    print("Query:", query)
    print("Top results:")

    for result in results:
        print()
        print("Document:", result["document_id"])
        print("Title:", result["title"])
        print("Score:", round(result["score"], 4))
        print("Text:", result["text"])