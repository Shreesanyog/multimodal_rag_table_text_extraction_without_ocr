from typing import List, Dict, Tuple, Any

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

try:
    from google.api_core.exceptions import ResourceExhausted, ServiceUnavailable, InternalServerError
except Exception:
    class ResourceExhausted(Exception):
        pass

    class ServiceUnavailable(Exception):
        pass

    class InternalServerError(Exception):
        pass

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

from config import GEMINI_CHAT_MODEL, RETRIEVAL_K, FINAL_CITATION_COUNT
from ingest import load_vectorstore
from hybrid_retriever import (
    exact_metadata_search,
    keyword_search_documents,
    bm25_search_documents,
    dedupe_docs,
    tokenize,
)


LAST_RAG_METRICS = {}


def get_llm(temperature: float = 0.0):
    return ChatGoogleGenerativeAI(
        model=GEMINI_CHAT_MODEL,
        temperature=temperature
    )


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=15),
    retry=retry_if_exception_type((ResourceExhausted, ServiceUnavailable, InternalServerError)),
)
def llm_text_invoke(prompt: str) -> str:
    llm = get_llm(temperature=0.0)
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content.strip()


def format_history(chat_history: List[Dict[str, str]], max_turns: int = 6) -> str:
    recent = chat_history[-max_turns:]
    return "\n".join([f"{m['role'].upper()}: {m['content']}" for m in recent])


def is_memory_question(question: str) -> bool:
    q = question.lower()
    patterns = [
        "what is my name",
        "who am i",
        "what did i tell",
        "what did i say",
        "remember my",
        "do you remember",
    ]
    return any(p in q for p in patterns)


def answer_from_memory(question: str, chat_history: List[Dict[str, str]]) -> str:
    history = format_history(chat_history, max_turns=15)

    prompt = f"""
You answer only from chat history.

Rules:
1. If the answer is available in chat history, answer directly.
2. If not available, say exactly:
   "I do not have that information in this chat."

Chat history:
{history}

Question:
{question}

Final answer:
"""
    return llm_text_invoke(prompt)


def faiss_search_documents(vectorstore, query: str):
    try:
        return vectorstore.max_marginal_relevance_search(
            query,
            k=RETRIEVAL_K,
            fetch_k=max(35, RETRIEVAL_K * 4),
            lambda_mult=0.55,
        )
    except Exception:
        return vectorstore.similarity_search(query, k=RETRIEVAL_K)


def retrieve_documents(question: str):
    vectorstore = load_vectorstore()

    if vectorstore is None:
        return [], {}

    exact_docs = exact_metadata_search(vectorstore, question)

    keyword_docs, keyword_debug = keyword_search_documents(
        vectorstore=vectorstore,
        question=question,
        top_k=14,
    )

    bm25_docs, bm25_debug = bm25_search_documents(
        vectorstore=vectorstore,
        question=question,
        top_k=14,
    )

    faiss_docs = faiss_search_documents(vectorstore, question)

    docs = dedupe_docs(exact_docs + keyword_docs + bm25_docs + faiss_docs)

    # Table chunks are useful for table questions
    q = question.lower()
    if "table" in q or "row" in q or "column" in q or "percent" in q or "policy" in q:
        docs = sorted(
            docs,
            key=lambda d: 0 if (d.metadata or {}).get("source_kind") in {"table", "table_row"} else 1
        )

    docs = docs[:RETRIEVAL_K]

    debug = {
        "exact_count": len(exact_docs),
        "keyword_count": len(keyword_docs),
        "bm25_count": len(bm25_docs),
        "faiss_count": len(faiss_docs),
        "bm25": bm25_debug,
        "keyword": keyword_debug,
    }

    return docs, debug

def top_unique_citations(docs):
    """
    Select citation docs for UI/answer.

    Rules:
    1. Show one citation per PDF page.
    2. If both page text and table chunk exist for same page, prefer table/table_row.
    3. Keep max FINAL_CITATION_COUNT citations.
    """

    best_by_page = {}

    def priority(doc):
        meta = doc.metadata or {}
        kind = meta.get("source_kind", "")

        # Prefer structured table evidence over plain page text
        if kind in {"table", "table_row"}:
            return 0

        return 1

    for doc in docs:
        meta = doc.metadata or {}

        key = (
            meta.get("file_name"),
            meta.get("page"),
        )

        if key not in best_by_page:
            best_by_page[key] = doc
        else:
            existing = best_by_page[key]
            if priority(doc) < priority(existing):
                best_by_page[key] = doc

    selected = list(best_by_page.values())

    return selected[:FINAL_CITATION_COUNT]
def build_text_context(docs) -> str:
    blocks = []

    for idx, doc in enumerate(docs, start=1):
        meta = doc.metadata or {}

        file_name = meta.get("file_name")
        page = meta.get("page")
        source_kind = meta.get("source_kind")
        table_number = meta.get("table_number")
        caption = meta.get("caption")

        label = f"Source file: {file_name}\nPage: {page}\nSource kind: {source_kind}"

        if table_number:
            label += f"\nTable: {table_number}"
        if caption:
            label += f"\nCaption: {caption}"

        blocks.append(f"""
[CONTEXT {idx}]
{label}

Content:
{(doc.page_content or '')[:5500]}
""")

    return "\n\n".join(blocks)


def ensure_inline_citation(answer: str, citation_docs) -> str:
    if not citation_docs:
        return answer

    if "(Source:" in answer:
        return answer

    top = citation_docs[0]
    meta = top.metadata or {}

    file_name = meta.get("file_name", "Unknown file")
    page = meta.get("page", "unknown")

    if meta.get("table_number"):
        return f"{answer.rstrip()} (Source: {file_name}, page {page}, Table {meta.get('table_number')})"

    return f"{answer.rstrip()} (Source: {file_name}, page {page})"


def compute_metrics(question: str, docs, citation_docs, retrieval_debug: dict):
    q_tokens = set(tokenize(question))
    ctx = " ".join([(d.page_content or "")[:2000] for d in docs[:4]])
    ctx_tokens = set(tokenize(ctx))

    coverage = 0.0
    if q_tokens:
        coverage = len(q_tokens.intersection(ctx_tokens)) / len(q_tokens)

    table_hits = sum(
        1 for d in docs
        if (d.metadata or {}).get("source_kind") in {"table", "table_row"}
    )

    return {
        "keyword_coverage": round(coverage, 3),
        "citation_count": len(citation_docs),
        "table_hits": table_hits,
        "bm25_used": retrieval_debug.get("bm25", {}).get("bm25_used", False),
        "bm25_available": retrieval_debug.get("bm25", {}).get("bm25_available", False),
        "top_keyword_matches": retrieval_debug.get("keyword", {}).get("top_keyword_matches", []),
        "retrieval_debug": retrieval_debug,
        "mode": "text_table_only_no_ocr",
        "note": "Only extractable PDF text and structured tables are indexed. Visual/OCR items are intentionally excluded.",
    }


def get_last_rag_metrics():
    return LAST_RAG_METRICS


def get_rag_answer(question: str, chat_history: List[Dict[str, str]]) -> Tuple[str, List[Any]]:
    global LAST_RAG_METRICS

    if is_memory_question(question):
        answer = answer_from_memory(question, chat_history)
        LAST_RAG_METRICS = {
            "mode": "chat_memory",
            "note": "Answered from chat memory, not PDF retrieval.",
        }
        return answer, []

    docs, retrieval_debug = retrieve_documents(question)

    if not docs:
        LAST_RAG_METRICS = {"mode": "no_docs"}
        return "I could not find this information in the uploaded documents.", []

    citation_docs = top_unique_citations(docs)
    context = build_text_context(docs)
    history = format_history(chat_history, max_turns=6)

    prompt = f"""
You are a strict document-grounded RAG assistant.

Architecture:
- The knowledge base contains only extractable PDF text and structured tables.
- OCR, figures, flowcharts, maps, screenshots and visual chart reading are intentionally excluded.
- Do not claim information from visual-only figures unless it appears in the retrieved text/table context.

Rules:
1. Use only the retrieved context.
2. Do not use outside knowledge.
3. If the answer is not present in the retrieved context, say:
   "I could not find this information in the uploaded documents."
4. Always include inline citations.
5. Citation format:
   (Source: exact_pdf_file_name, page exact_page_number)
   or, for tables:
   (Source: exact_pdf_file_name, page exact_page_number, Table exact_table_number)
6. If answering from a table, mention the row/column meaning clearly.
7. Do not invent numeric values.
8. If the question asks about a figure, chart, color box, diagram or visual layout, explain that this version does not index OCR/visual content unless the answer is present in extracted text.

Chat history:
{history}

Retrieved context:
{context}

User question:
{question}

Final answer:
"""

    try:
        answer = llm_text_invoke(prompt)
    except Exception as e:
        print(f"LLM answer failed: {e}")
        answer = "I could not generate an answer from the retrieved documents."

    answer = ensure_inline_citation(answer, citation_docs)

    LAST_RAG_METRICS = compute_metrics(
        question=question,
        docs=docs,
        citation_docs=citation_docs,
        retrieval_debug=retrieval_debug,
    )

    return answer, citation_docs