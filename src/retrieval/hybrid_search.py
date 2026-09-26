from typing import List, Dict, Any
from rank_bm25 import BM25Okapi

class HybridRetriever:
    def __init__(self, documents: List[Dict[str, Any]]):
        self.documents = documents
        tokenized_corpus = [doc["text"].lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def search_bm25(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [{"doc": self.documents[i], "score": float(scores[i])} for i in top_indices]

    def combine_rrf(
        self,
        dense_results: List[Dict[str, Any]],
        sparse_results: List[Dict[str, Any]],
        k: int = 60,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        rrf_scores: Dict[str, float] = {}
        doc_map: Dict[str, Dict[str, Any]] = {}

        for rank, item in enumerate(dense_results):
            doc_id = item["doc"]["id"]
            doc_map[doc_id] = item["doc"]
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k + rank + 1))

        for rank, item in enumerate(sparse_results):
            doc_id = item["doc"]["id"]
            doc_map[doc_id] = item["doc"]
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k + rank + 1))

        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [{"doc": doc_map[doc_id], "rrf_score": score} for doc_id, score in sorted_docs]