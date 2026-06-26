import re
from collections import defaultdict

from rank_bm25 import BM25Okapi

from app.config import TOP_K_DENSE, TOP_K_FINAL, TOP_K_SPARSE
from app.storage.faiss_store import ChunkRecord, vector_store


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


class HybridRetriever:
    def __init__(self):
        self._bm25: BM25Okapi | None = None
        self._corpus: list[list[str]] = []

    def _ensure_bm25(self) -> None:
        if not vector_store.records:
            self._bm25 = None
            self._corpus = []
            return
        self._corpus = [_tokenize(record.text) for record in vector_store.records]
        self._bm25 = BM25Okapi(self._corpus)

    def sparse_search(self, query: str, top_k: int) -> list[tuple[ChunkRecord, float]]:
        self._ensure_bm25()
        if not self._bm25:
            return []
        tokens = _tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(
            enumerate(scores),
            key=lambda item: item[1],
            reverse=True,
        )[:top_k]
        return [
            (vector_store.records[idx], float(score))
            for idx, score in ranked
            if score > 0
        ]

    def retrieve(self, query: str) -> list[ChunkRecord]:
        if not vector_store.is_ready:
            return []

        dense = vector_store.dense_search(query, TOP_K_DENSE)
        sparse = self.sparse_search(query, TOP_K_SPARSE)

        combined: dict[str, tuple[ChunkRecord, float]] = {}
        for record, score in dense:
            key = f"{record.source}:{record.chunk_id}"
            combined[key] = (record, score * 0.65)

        max_sparse = max((score for _, score in sparse), default=1.0) or 1.0
        for record, score in sparse:
            key = f"{record.source}:{record.chunk_id}"
            normalized = (score / max_sparse) * 0.35
            if key in combined:
                existing_record, existing_score = combined[key]
                combined[key] = (existing_record, existing_score + normalized)
            else:
                combined[key] = (record, normalized)

        ranked = sorted(combined.values(), key=lambda item: item[1], reverse=True)
        return [record for record, _ in ranked[:TOP_K_FINAL]]


hybrid_retriever = HybridRetriever()
