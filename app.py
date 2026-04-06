from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from src.generation import (
    DEFAULT_GEMINI_MODEL,
    LLM_ASSISTED_MODE,
    STANDALONE_RAG_MODE,
    generate_answer,
)
from src.indexing import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_EMBEDDING_MODEL,
    build_index_from_pdf,
)
from src.retrieval import DEFAULT_TOP_K, retrieve_chunks


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_PDF_PATH = PROJECT_ROOT / "data" / "document.pdf"


st.set_page_config(page_title="RAG Concept Demo", page_icon=":books:", layout="wide")

st.title("RAG Concept Demo")
st.caption("A simple Retrieval-Augmented Generation pipeline for technical documents.")


def build_bundle_from_uploaded_file(uploaded_file, chunk_size: int, chunk_overlap: int):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        temp_path = Path(tmp_file.name)
    try:
        return build_index_from_pdf(
            temp_path,
            model_name=DEFAULT_EMBEDDING_MODEL,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            source_name=uploaded_file.name,
        )
    finally:
        temp_path.unlink(missing_ok=True)


def reset_bundle_state() -> None:
    st.session_state.bundle = None
    st.session_state.bundle_source = None
    st.session_state.pipeline_summary = None
    st.session_state.last_query_summary = None


if "bundle" not in st.session_state:
    st.session_state.bundle = None
if "bundle_source" not in st.session_state:
    st.session_state.bundle_source = None
if "pipeline_summary" not in st.session_state:
    st.session_state.pipeline_summary = None
if "show_api_key_input" not in st.session_state:
    st.session_state.show_api_key_input = False
if "gemini_api_key" not in st.session_state:
    st.session_state.gemini_api_key = ""
if "last_query_summary" not in st.session_state:
    st.session_state.last_query_summary = None


with st.sidebar:
    st.header("Settings")
    ingestion_mode = st.radio(
        "Document source",
        ("Default bundled PDF", "Upload a PDF"),
    )
    rag_mode_label = st.selectbox(
        "Answer mode",
        ("Pure / standalone RAG", "LLM-assisted RAG"),
    )
    chunk_size = st.slider("Chunk size", min_value=200, max_value=500, value=DEFAULT_CHUNK_SIZE, step=25)
    chunk_overlap = st.slider("Chunk overlap", min_value=0, max_value=150, value=DEFAULT_CHUNK_OVERLAP, step=10)
    top_k = st.slider("Top-k retrieval", min_value=1, max_value=10, value=DEFAULT_TOP_K)
    if st.button("Enter Gemini API key"):
        st.session_state.show_api_key_input = not st.session_state.show_api_key_input
    if st.session_state.show_api_key_input:
        st.session_state.gemini_api_key = st.text_input(
            "Gemini API key",
            value=st.session_state.gemini_api_key,
            type="password",
            help="Stored only in this Streamlit session.",
        )
    st.caption("These defaults are adjustable so you can experiment with retrieval behavior.")


rag_mode = (
    STANDALONE_RAG_MODE
    if rag_mode_label == "Pure / standalone RAG"
    else LLM_ASSISTED_MODE
)


uploaded_pdf = None
if ingestion_mode == "Upload a PDF":
    uploaded_pdf = st.file_uploader("Choose a PDF", type=["pdf"])
else:
    st.info(f"Using bundled default document: `data/{DEFAULT_PDF_PATH.name}`")


st.info(
    "Pipeline: 1) extract PDF text, 2) chunk with overlap, 3) embed with "
    f"`{DEFAULT_EMBEDDING_MODEL}`, 4) retrieve top-{top_k} chunks, 5) answer in "
    f"`{rag_mode_label}`."
)

model_lines = [
    f"Embedding model: `{DEFAULT_EMBEDDING_MODEL}`",
    f"Answer mode: `{rag_mode_label}`",
]
if rag_mode == LLM_ASSISTED_MODE:
    key_status = "configured in this session" if st.session_state.gemini_api_key else "not configured"
    model_lines.append(f"LLM model: `{DEFAULT_GEMINI_MODEL}`")
    model_lines.append(f"Gemini API key: {key_status}")
else:
    model_lines.append("LLM model: not used in pure / standalone RAG mode")

st.caption(" | ".join(model_lines))


if st.button("Build or refresh index", type="primary"):
    try:
        if ingestion_mode == "Default bundled PDF":
            if not DEFAULT_PDF_PATH.exists():
                raise FileNotFoundError(
                    f"Default bundled PDF not found at {DEFAULT_PDF_PATH}"
                )
            st.session_state.bundle = build_index_from_pdf(
                DEFAULT_PDF_PATH,
                model_name=DEFAULT_EMBEDDING_MODEL,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                source_name=f"data/{DEFAULT_PDF_PATH.name}",
            )
            st.session_state.bundle_source = f"data/{DEFAULT_PDF_PATH.name}"
        else:
            if uploaded_pdf is None:
                raise ValueError("Please upload a PDF before building the index.")
            st.session_state.bundle = build_bundle_from_uploaded_file(
                uploaded_pdf, chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
            st.session_state.bundle_source = uploaded_pdf.name
        st.session_state.last_query_summary = None
        st.session_state.pipeline_summary = {
            "source": st.session_state.bundle_source,
            "source_path": (
                f"data/{DEFAULT_PDF_PATH.name}"
                if ingestion_mode == "Default bundled PDF"
                else st.session_state.bundle_source
            ),
            "pages": st.session_state.bundle.page_count,
            "chunks": len(st.session_state.bundle.chunks),
            "chunk_size": st.session_state.bundle.chunk_size,
            "chunk_overlap": st.session_state.bundle.chunk_overlap,
            "embedding_model": st.session_state.bundle.model_name,
        }
        st.success(f"Indexed document: {st.session_state.bundle_source}")
    except Exception as exc:
        reset_bundle_state()
        st.error(f"Indexing failed: {exc}")

if st.session_state.pipeline_summary:
    summary = st.session_state.pipeline_summary
    st.success(
        "Document processed successfully: "
        f"{summary['source']} | {summary['pages']} page(s) | {summary['chunks']} chunk(s)"
    )
    st.caption(
        f"Source: {summary['source_path']} | "
        "Index settings: "
        f"chunk size={summary['chunk_size']}, overlap={summary['chunk_overlap']}, "
        f"embedding model={summary['embedding_model']}"
    )

query = st.text_input("Ask a question about the indexed document")

if st.button("Run retrieval and generation"):
    if st.session_state.bundle is None:
        st.warning("Build the index first.")
    elif not query.strip():
        st.warning("Enter a question before running the pipeline.")
    else:
        try:
            st.info("Running retrieval over the indexed PDF and preparing the answer...")
            retrieved = retrieve_chunks(query, st.session_state.bundle, k=top_k)
            response = generate_answer(
                query,
                retrieved,
                mode=rag_mode,
                model_name=DEFAULT_GEMINI_MODEL,
                api_key=st.session_state.gemini_api_key or None,
            )
            st.session_state.last_query_summary = {
                "query": query,
                "results": len(retrieved),
                "top_score": retrieved[0].score if retrieved else None,
                "mode": response["mode"],
            }

            left, right = st.columns([3, 2])
            with left:
                st.subheader("Grounded answer")
                if response["mode"] == LLM_ASSISTED_MODE:
                    st.success(response["status_message"])
                elif response["mode"].startswith("fallback") or response["mode"] == "insufficient-evidence":
                    st.warning(response["status_message"])
                else:
                    st.info(response["status_message"])
                st.write(response["answer"])
                st.caption(f"Generation mode: {response['mode']}")
                if rag_mode == LLM_ASSISTED_MODE:
                    st.caption(f"Configured LLM: `{DEFAULT_GEMINI_MODEL}`")
                else:
                    st.caption("Configured mode: standalone RAG without Gemini synthesis")
                st.caption(
                    "Query workflow: vectorize the question, retrieve the highest-scoring chunks from FAISS, "
                    "then answer using either standalone evidence formatting or Gemini-assisted synthesis."
                )

            with right:
                st.subheader("Retrieved evidence")
                for item in retrieved:
                    citation = (
                        f"rank={item.rank} | {item.chunk_id} | score={item.score:.3f} | page={item.page_number}"
                    )
                    with st.expander(citation):
                        st.write(item.text)
        except Exception as exc:
            st.error(f"Question answering failed: {exc}")

if st.session_state.last_query_summary:
    query_summary = st.session_state.last_query_summary
    top_score_text = (
        f"{query_summary['top_score']:.3f}"
        if query_summary["top_score"] is not None
        else "n/a"
    )
    st.caption(
        "Last query summary: "
        f"{query_summary['results']} retrieved chunk(s) | "
        f"top similarity={top_score_text} | "
        f"response mode={query_summary['mode']}"
    )
