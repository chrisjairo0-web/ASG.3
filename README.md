# RAG Concept Demo

This project is a small Retrieval-Augmented Generation (RAG) demo for technical PDFs. It supports both ingestion modes requested in the prompt:

- a default bundled PDF at `data/document.pdf` for an offline demo path
- a user-uploaded PDF for an interactive path

The app chunks document text, embeds each chunk with `sentence-transformers`, stores vectors in FAISS, retrieves the most relevant chunks for a question, and then produces a grounded answer with citations.

## Schematic Overview

```text
                       +----------------------+
                       |   Streamlit UI       |
                       |      app.py          |
                       +----------+-----------+
                                  |
                 +----------------+----------------+
                 |                                 |
                 v                                 v
      +----------------------+         +----------------------+
      | Default document     |         | Uploaded PDF         |
      | data/document.pdf    |         | st.file_uploader     |
      +----------+-----------+         +----------+-----------+
                 |                                 |
                 +----------------+----------------+
                                  |
                                  v
                      +--------------------------+
                      | Indexing Pipeline        |
                      | src/indexing.py          |
                      | - PDF text extraction    |
                      | - chunking with overlap  |
                      | - sentence embeddings    |
                      | - FAISS index build      |
                      +------------+-------------+
                                   |
                                   v
                      +--------------------------+
                      | Indexed Bundle           |
                      | - chunks + metadata      |
                      | - embedding model name   |
                      | - FAISS vector index     |
                      +------------+-------------+
                                   |
                         user query |
                                   v
                      +--------------------------+
                      | Retrieval                |
                      | src/retrieval.py         |
                      | - embed query            |
                      | - top-k similarity       |
                      | - ranked evidence        |
                      +------------+-------------+
                                   |
                                   v
                      +--------------------------+
                      | Generation               |
                      | src/generation.py        |
                      | - standalone RAG         |
                      | - Gemini-assisted RAG    |
                      +------------+-------------+
                                   |
                                   v
                      +--------------------------+
                      | Answer + citations       |
                      | Retrieved evidence view  |
                      +--------------------------+
```

## Project Structure

```text
.
|-- app.py
|-- data/
|   `-- document.pdf
|-- prompt_rag.md
|-- requirements.txt
`-- src/
    |-- generation.py
    |-- indexing.py
    `-- retrieval.py
```

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Optional: set `GEMINI_API_KEY` if you want Gemini-assisted answers instead of the built-in standalone RAG output.

## Run

```bash
streamlit run app.py
```

## How It Works

### 1. Indexing

`src/indexing.py` loads a PDF, extracts text page by page, creates overlapping chunks, embeds them, and builds a FAISS similarity index. Chunk metadata includes:

- source
- page number
- chunk ID

Configurable defaults:

- chunk size: `300`
- chunk overlap: `50`
- embedding model: `sentence-transformers/all-MiniLM-L6-v2`

The default `data/document.pdf` file goes through this same extraction, chunking, embedding, and indexing pipeline when you click `Build or refresh index`.

### 2. Retrieval

`src/retrieval.py` embeds the user query with the same embedding model and performs top-k similarity search over the FAISS index.

Configurable default:

- top-k retrieval: `5`

This is a starter value, not a universally optimal one. You can tune it based on corpus size, latency, and answer quality.

### 3. Generation

`src/generation.py` supports two answer modes:

- pure / standalone RAG: answer directly from the retrieved chunks and citations without Gemini synthesis
- LLM-assisted RAG: send the augmented prompt to Gemini through the official Google GenAI SDK when a `GEMINI_API_KEY` is available

If Gemini mode is selected without an API key, the app falls back to standalone RAG and explains why.

If retrieval returns weak or no evidence, the generation layer is designed to avoid inventing facts and say that confidence is limited.

## End-to-End Flow

1. The user chooses either the bundled `data/document.pdf` file or uploads a PDF.
2. The app runs the indexing pipeline to extract text, chunk it, embed it, and build a FAISS index.
3. The user enters a question.
4. The retrieval layer embeds the question and returns the highest-ranked chunks.
5. The generation layer either:
   uses standalone RAG to format the retrieved evidence directly, or
   sends the augmented prompt to Gemini for synthesis.
6. The UI displays the answer, citations, retrieved evidence, and status messages about the active mode.

## Ingestion Modes

### Bundled Default PDF

Use the bundled file `data/document.pdf` to try the app immediately.

### Uploaded PDF

Choose `Upload a PDF` in the sidebar and then build a fresh index from the uploaded file.

## App Controls

- `Build or refresh index` processes the selected PDF through extraction, chunking, embedding, and FAISS indexing
- `Enter Gemini API key` reveals a Gemini API key field stored only in the current Streamlit session
- `Answer mode` lets you choose between pure / standalone RAG and Gemini-assisted RAG
- The app shows informative status messages for the active embedding model, answer mode, Gemini model, and last query summary

## Notes

- Chunking uses whitespace token approximation to keep the demo simple and transparent.
- The sidebar shows the embedding model, answer mode, and Gemini model status so the active pipeline is visible.
- The app includes a button that reveals a session-local Gemini API key input.
- The Gemini call is optional, so the demo remains runnable without external API access.
- If Gemini import fails with an error mentioning `genai` and `google`, verify that `google-genai` is installed in the same environment running Streamlit. Older `google` or `google-generativeai` packages can also conflict with the `google.genai` import path.
- FAISS metadata is kept in memory in the Streamlit flow, but `src/indexing.py` also includes helpers to save and reload the index bundle.
