import chromadb
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv
import re

load_dotenv()

CHROMA_PATH = "chroma_store"
COLLECTION_NAME = "documents"
TOP_K_SEMANTIC = 20
TOP_K_BM25 = 20
RRF_K = 60

def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_collection(name=COLLECTION_NAME)


def tokenize(text: str) -> list[str]:
    return re.findall(r'\w+', text.lower())


def semantic_search(query: str, collection, top_k: int = TOP_K_SEMANTIC) -> list[dict]:
    embedder = HuggingFaceEmbeddings()
    query_embeddings = embedder.embed_query(query)

    results = collection.query(
        query_embeddings = [query_embeddings],
        n_results = top_k,
        include = ["documents", "metadatas", "distances"]
    )

    hits = []
    for i, doc_id in enumerate(results["ids"][0]):
        hits.append({
            "id": doc_id,
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "score": 1 - results["distances"][0][i],
            "rank": i + 1
        })
    return hits


def bm25_search(query: str, collection, top_k: int = TOP_K_BM25) -> list[dict]:
    all_data = collection.get(include = ["documents", "metadatas"])

    if not all_data["documents"]:
        return []
    
    all_docs = all_data["documents"]
    all_ids = all_data["ids"]
    all_metas = all_data["metadatas"]

    tokenized_corpus = [tokenize(doc) for doc in all_docs]
    bm25 = BM25Okapi(tokenized_corpus)

    tokenized_query = tokenize(query)
    scores = bm25.get_scores(tokenized_query)

    ranked_indices = sorted(
        range(len(scores)),
        key = lambda i : scores[i],
        reverse = True
    )[:top_k]

    hits = []
    for rank, idx in enumerate(ranked_indices):
        hits.append({
            "id": all_ids[idx],
            "text": all_docs[idx],
            "metadata": all_metas[idx],
            "score": float(scores[idx]),
            "rank": rank + 1
        })
    return hits

def reciprocal_rank_fusion(semantic_hits: list[dict], bm25_hits: list[dict], k: int = RRF_K) -> list[dict]:
    doc_map = {}
    for hit in semantic_hits + bm25_hits:
        if hit["id"] not in doc_map:
            doc_map[hit["id"]] = {
                "id": hit["id"],
                "text": hit["text"],
                "metadata": hit["metadata"],
                "rrf_score": 0.0
            }

    for hit in semantic_hits:
        doc_map[hit["id"]]["rrf_score"] += 1.0 / (k + hit["rank"]);

    for hit in bm25_hits:
        doc_map[hit["id"]]["rrf_score"] += 1.0 / (k + hit["rank"]);

    merged = sorted(doc_map.values(), key = lambda x: x["rrf_score"], reverse = True)
    return merged


def hybrid_search(query: str, top_k_final: int = 20) -> list[dict]:
    collection = get_collection()
    semantic_hits = semantic_search(query, collection)
    bm25_hits = bm25_search(query, collection)
    merged = reciprocal_rank_fusion(semantic_hits, bm25_hits)
    return merged[:top_k_final]


if __name__ == "__main__":
    query = "What is the main topic of the document?"
    results = hybrid_search(query, top_k_final=5)
    for i, r in enumerate(results, 1):
        print(f"\n--- Result {i} (RRF: {r['rrf_score']:.4f}) ---")
        print(f"Source: {r['metadata']['doc_name']}, Page {r['metadata']['page_number']}")
        print(r['text'][:200])