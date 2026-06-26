import pickle
from dataclasses import dataclass, field

import faiss
import numpy as np
from openai import OpenAI

from app.config import (
    FAISS_INDEX_PATH,
    METADATA_PATH,
    OPENAI_API_KEY,
    OPENAI_EMBEDDING_MODEL,
)


@dataclass
class ChunkRecord:
    text: str
    source: str
    chunk_id: int


@dataclass
class VectorStore:
    index: faiss.IndexFlatIP | None = None
    records: list[ChunkRecord] = field(default_factory=list)
    embeddings: np.ndarray | None = None

    def __post_init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.load()

    @property
    def is_ready(self) -> bool:
        return self.index is not None and len(self.records) > 0

    def load(self) -> None:
        if FAISS_INDEX_PATH.exists() and METADATA_PATH.exists():
            self.index = faiss.read_index(str(FAISS_INDEX_PATH))
            with open(METADATA_PATH, "rb") as f:
                data = pickle.load(f)
            self.records = data["records"]
            self.embeddings = data.get("embeddings")

    def save(self) -> None:
        if self.index is None:
            return
        faiss.write_index(self.index, str(FAISS_INDEX_PATH))
        with open(METADATA_PATH, "wb") as f:
            pickle.dump(
                {"records": self.records, "embeddings": self.embeddings},
                f,
            )

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        response = self.client.embeddings.create(
            model=OPENAI_EMBEDDING_MODEL,
            input=texts,
        )
        vectors = np.array(
            [item.embedding for item in response.data],
            dtype=np.float32,
        )
        faiss.normalize_L2(vectors)
        return vectors

    def add_documents(self, chunks: list[tuple[str, str]]) -> int:
        if not chunks:
            return 0
        texts = [text for text, _ in chunks]
        vectors = self.embed_texts(texts)
        new_records = [
            ChunkRecord(text=text, source=source, chunk_id=i)
            for i, (text, source) in enumerate(chunks)
        ]
        if self.index is None:
            dim = vectors.shape[1]
            self.index = faiss.IndexFlatIP(dim)
            self.records = []
            self.embeddings = None

        self.index.add(vectors)
        self.records.extend(new_records)
        if self.embeddings is None or self.embeddings.size == 0:
            self.embeddings = vectors
        else:
            self.embeddings = np.vstack([self.embeddings, vectors])
        self.save()
        return len(chunks)

    def dense_search(self, query: str, top_k: int) -> list[tuple[ChunkRecord, float]]:
        if not self.is_ready:
            return []
        query_vec = self.embed_texts([query])
        scores, indices = self.index.search(query_vec, min(top_k, len(self.records)))
        results: list[tuple[ChunkRecord, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            results.append((self.records[idx], float(score)))
        return results

    def clear(self) -> None:
        self.index = None
        self.records = []
        self.embeddings = None
        if FAISS_INDEX_PATH.exists():
            FAISS_INDEX_PATH.unlink()
        if METADATA_PATH.exists():
            METADATA_PATH.unlink()

    def get_sources(self) -> list[str]:
        return sorted({record.source for record in self.records})

    def get_documents(self) -> list[dict[str, str | int]]:
        counts: dict[str, int] = {}
        for record in self.records:
            counts[record.source] = counts.get(record.source, 0) + 1
        return [
            {"filename": name, "chunks": counts[name]}
            for name in sorted(counts)
        ]

    def remove_source(self, source: str) -> int:
        if not self.records:
            return 0

        keep_indices = [i for i, record in enumerate(self.records) if record.source != source]
        removed = len(self.records) - len(keep_indices)
        if removed == 0:
            return 0

        if not keep_indices:
            self.clear()
            return removed

        self.records = [self.records[i] for i in keep_indices]
        if self.embeddings is not None:
            self.embeddings = self.embeddings[keep_indices]
            dim = self.embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dim)
            self.index.add(self.embeddings)
        self.save()
        return removed

    def sample_context(self, max_chars: int = 4000) -> str:
        if not self.records:
            return ""
        parts: list[str] = []
        total = 0
        for record in self.records:
            snippet = record.text[:500]
            if total + len(snippet) > max_chars:
                break
            parts.append(f"[{record.source}] {snippet}")
            total += len(snippet)
        return "\n".join(parts)


vector_store = VectorStore()
