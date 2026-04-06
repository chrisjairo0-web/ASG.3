from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_CHUNK_SIZE = 300
DEFAULT_CHUNK_OVERLAP = 50


@dataclass
class ChunkRecord:
    chunk_id: str
    source: str
    page_number: int | None
    text: str


@dataclass
class IndexBundle:
    index: faiss.Index
    chunks: list[ChunkRecord]
    model_name: str
    chunk_size: int
    chunk_overlap: int
    page_count: int
    source_name: str


def load_pdf_pages(pdf_path: str | Path) -> list[tuple[int, str]]:
    """Extract page text from a PDF."""
    path = Path(pdf_path)
    reader = PdfReader(str(path))
    pages: list[tuple[int, str]] = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append((page_number, normalize_whitespace(text)))

    if not pages:
        raise ValueError(f"No extractable text found in PDF: {path}")

    return pages


def normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def chunk_pages(
    pages: Iterable[tuple[int, str]],
    source_name: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[ChunkRecord]:
    """Create approximate token chunks with overlap from page text."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunks: list[ChunkRecord] = []

    for page_number, page_text in pages:
        tokens = page_text.split()
        if not tokens:
            continue

        start = 0
        local_chunk_index = 0
        step = chunk_size - chunk_overlap
        while start < len(tokens):
            token_slice = tokens[start : start + chunk_size]
            if not token_slice:
                break
            local_chunk_index += 1
            chunks.append(
                ChunkRecord(
                    chunk_id=f"page-{page_number}-chunk-{local_chunk_index}",
                    source=source_name,
                    page_number=page_number,
                    text=" ".join(token_slice),
                )
            )
            start += step

    if not chunks:
        raise ValueError("No chunks were created from the PDF text")

    return chunks


def get_embedding_model(model_name: str = DEFAULT_EMBEDDING_MODEL) -> SentenceTransformer:
    return SentenceTransformer(model_name)


def embed_chunks(
    chunks: list[ChunkRecord], model: SentenceTransformer
) -> np.ndarray:
    embeddings = model.encode(
        [chunk.text for chunk in chunks],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return embeddings.astype("float32")


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    return index


def build_index_from_pdf(
    pdf_path: str | Path,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    source_name: str | None = None,
) -> IndexBundle:
    path = Path(pdf_path)
    pages = load_pdf_pages(path)
    resolved_source_name = source_name or path.name
    chunks = chunk_pages(
        pages,
        source_name=resolved_source_name,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    model = get_embedding_model(model_name)
    embeddings = embed_chunks(chunks, model)
    index = build_faiss_index(embeddings)
    return IndexBundle(
        index=index,
        chunks=chunks,
        model_name=model_name,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        page_count=len(pages),
        source_name=resolved_source_name,
    )


def save_index_bundle(bundle: IndexBundle, output_dir: str | Path) -> None:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)

    faiss.write_index(bundle.index, str(path / "index.faiss"))
    metadata = {
        "model_name": bundle.model_name,
        "chunk_size": bundle.chunk_size,
        "chunk_overlap": bundle.chunk_overlap,
        "page_count": bundle.page_count,
        "source_name": bundle.source_name,
        "chunks": [asdict(chunk) for chunk in bundle.chunks],
    }
    (path / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def load_index_bundle(index_dir: str | Path) -> IndexBundle:
    path = Path(index_dir)
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    chunks = [ChunkRecord(**chunk) for chunk in metadata["chunks"]]
    index = faiss.read_index(str(path / "index.faiss"))
    return IndexBundle(
        index=index,
        chunks=chunks,
        model_name=metadata["model_name"],
        chunk_size=metadata["chunk_size"],
        chunk_overlap=metadata["chunk_overlap"],
        page_count=metadata.get("page_count", 0),
        source_name=metadata.get("source_name", "unknown.pdf"),
    )
