from __future__ import annotations

from dataclasses import dataclass

from sentence_transformers import SentenceTransformer

from src.indexing import IndexBundle, get_embedding_model


DEFAULT_TOP_K = 5


@dataclass
class RetrievalResult:
    rank: int
    score: float
    chunk_id: str
    source: str
    page_number: int | None
    text: str


def retrieve_chunks(
    query: str,
    bundle: IndexBundle,
    k: int = DEFAULT_TOP_K,
    model: SentenceTransformer | None = None,
) -> list[RetrievalResult]:
    if not query.strip():
        raise ValueError("query must not be empty")
    if k <= 0:
        raise ValueError("k must be greater than 0")

    embedding_model = model or get_embedding_model(bundle.model_name)
    query_embedding = embedding_model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")

    top_k = min(k, len(bundle.chunks))
    scores, indices = bundle.index.search(query_embedding, top_k)

    results: list[RetrievalResult] = []
    for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
        if idx < 0:
            continue
        chunk = bundle.chunks[idx]
        results.append(
            RetrievalResult(
                rank=rank,
                score=float(score),
                chunk_id=chunk.chunk_id,
                source=chunk.source,
                page_number=chunk.page_number,
                text=chunk.text,
            )
        )
    return results
