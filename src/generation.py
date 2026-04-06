from __future__ import annotations

import importlib
import os
from typing import Iterable

from src.retrieval import RetrievalResult


DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
STANDALONE_RAG_MODE = "standalone_rag"
LLM_ASSISTED_MODE = "llm_assisted"


def build_augmented_prompt(query: str, retrieved_chunks: Iterable[RetrievalResult]) -> str:
    context_blocks = []
    for item in retrieved_chunks:
        citation = format_citation(item)
        context_blocks.append(f"[{citation}]\n{item.text}")

    joined_context = "\n\n".join(context_blocks) if context_blocks else "No supporting context retrieved."
    return (
        "You are answering a technical question using retrieved evidence from an engineering document.\n"
        "Use only the retrieved evidence below. If the evidence is insufficient, say so clearly.\n\n"
        f"Question:\n{query}\n\n"
        f"Retrieved evidence:\n{joined_context}\n\n"
        "Answer with concise reasoning and cite the supporting chunk IDs."
    )


def format_citation(item: RetrievalResult) -> str:
    page = f", page {item.page_number}" if item.page_number is not None else ""
    return f"{item.source}{page}, {item.chunk_id}"


def generate_answer(
    query: str,
    retrieved_chunks: list[RetrievalResult],
    mode: str = LLM_ASSISTED_MODE,
    model_name: str = DEFAULT_GEMINI_MODEL,
    api_key: str | None = None,
) -> dict[str, str]:
    prompt = build_augmented_prompt(query, retrieved_chunks)

    if not retrieved_chunks:
        return {
            "answer": "I could not find relevant evidence in the indexed document, so I cannot answer confidently.",
            "prompt": prompt,
            "mode": "insufficient-evidence",
            "status_message": "No relevant evidence was retrieved from the indexed document.",
        }

    if mode == STANDALONE_RAG_MODE:
        standalone = build_standalone_rag_answer(query, retrieved_chunks)
        standalone["prompt"] = prompt
        standalone["mode"] = STANDALONE_RAG_MODE
        return standalone

    effective_api_key = api_key or os.getenv("GEMINI_API_KEY")
    if effective_api_key:
        try:
            genai = import_genai_module()

            client = genai.Client(api_key=effective_api_key)
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
            )
            return {
                "answer": response.text.strip(),
                "prompt": prompt,
                "mode": LLM_ASSISTED_MODE,
                "status_message": f"Gemini answered with model `{model_name}`.",
            }
        except Exception as exc:  # pragma: no cover - best-effort fallback
            fallback = build_standalone_rag_answer(query, retrieved_chunks)
            fallback["answer"] += f"\n\n[Fallback note: Gemini call failed with: {exc}]"
            fallback["prompt"] = prompt
            fallback["mode"] = "fallback-after-error"
            fallback["status_message"] = f"Gemini call failed, so the app fell back to standalone RAG: {exc}"
            return fallback

    fallback = build_standalone_rag_answer(query, retrieved_chunks)
    fallback["answer"] += (
        "\n\n[Mode note: Gemini assistance is selected, but no API key is configured. "
        "Showing standalone RAG output instead.]"
    )
    fallback["prompt"] = prompt
    fallback["mode"] = "fallback-no-api-key"
    fallback["status_message"] = (
        "LLM-assisted mode is selected, but no Gemini API key is available. "
        "The app fell back to standalone RAG."
    )
    return fallback


def build_standalone_rag_answer(
    query: str, retrieved_chunks: list[RetrievalResult]
) -> dict[str, str]:
    answer_lines = [
        f"Question: {query}",
        "",
        "Standalone RAG answer:",
        "The answer below is composed directly from the highest-ranked retrieved evidence, without LLM synthesis.",
        "",
        "Evidence-based response:",
    ]

    for item in retrieved_chunks[:3]:
        excerpt = compress_excerpt(item.text)
        answer_lines.append(
            f"- Rank {item.rank}: {excerpt} ({format_citation(item)})"
        )

    answer_lines.append("")
    answer_lines.append("Supporting citations:")
    for item in retrieved_chunks[:3]:
        answer_lines.append(f"- {format_citation(item)}")

    top_score = retrieved_chunks[0].score
    if top_score < 0.20:
        answer_lines.append("")
        answer_lines.append(
            "Confidence note: the similarity scores are low, so the retrieved evidence may not fully answer the question."
        )
    return {
        "answer": "\n".join(answer_lines),
        "status_message": "Standalone RAG answered directly from retrieved evidence.",
    }


def compress_excerpt(text: str, max_words: int = 45) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + " ..."


def import_genai_module():
    try:
        return importlib.import_module("google.genai")
    except ImportError as first_error:
        try:
            from google import genai

            return genai
        except ImportError as second_error:
            raise ImportError(
                "Could not import the Google GenAI SDK. "
                "Make sure `google-genai` is installed in the active environment. "
                "If you also installed the legacy `google-generativeai` package or a plain `google` package, "
                "they may be conflicting with `google.genai`."
            ) from second_error
