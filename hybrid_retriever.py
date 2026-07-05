import re
import math
from typing import List


try:
    from rank_bm25 import BM25Okapi
    RANK_BM25_AVAILABLE = True
except Exception:
    BM25Okapi = None
    RANK_BM25_AVAILABLE = False


STOPWORDS = {
    "the", "is", "are", "a", "an", "and", "or", "of", "to", "in", "on",
    "for", "with", "which", "what", "who", "whom", "whose", "that",
    "this", "these", "those", "name", "give", "tell", "list", "show",
    "me", "from", "pdf", "page", "figure", "table", "please", "can",
    "you", "answer", "according", "based", "inside", "under", "does",
    "say", "written", "about", "it", "as", "by", "at", "be", "was",
    "were", "will", "would", "should", "could", "into", "its", "their",
    "there", "here", "also", "then", "than", "only", "all", "any"
}


def normalize_text(text: str) -> str:
    """
    Normalize text for keyword and BM25 matching.
    """
    text = (text or "").lower()
    text = text.replace("‑", "-").replace("–", "-").replace("—", "-")
    text = re.sub(r"[^a-z0-9%.\- ]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize(text: str) -> List[str]:
    """
    Tokenize text for keyword and BM25 retrieval.
    """
    norm = normalize_text(text)
    tokens = []

    for token in norm.split():
        token = token.strip()

        if len(token) <= 2:
            continue

        if token in STOPWORDS:
            continue

        tokens.append(token)

    return tokens


def get_all_vector_docs(vectorstore):
    """
    Pull all stored documents from LangChain FAISS docstore.
    """
    try:
        return list(vectorstore.docstore._dict.values())
    except Exception:
        return []


def exact_metadata_search(vectorstore, question: str):
    """
    Exact matching for:
    - PDF file name
    - page number
    - table number

    Useful when user asks:
    "In Table 3 of 2025_pdf.pdf..."
    """
    all_docs = get_all_vector_docs(vectorstore)
    matched = []

    file_match = re.search(
        r"([a-zA-Z0-9_\- ]+\.pdf)",
        question,
        flags=re.IGNORECASE,
    )

    page_match = re.search(
        r"\bpage\s*(\d+)\b",
        question,
        flags=re.IGNORECASE,
    )

    table_match = re.search(
        r"\btable\s*(\d+)\b",
        question,
        flags=re.IGNORECASE,
    )

    requested_file = file_match.group(1).strip().lower() if file_match else None
    requested_page = int(page_match.group(1)) if page_match else None
    requested_table = table_match.group(1) if table_match else None

    if not requested_file and requested_page is None and not requested_table:
        return []

    for doc in all_docs:
        meta = doc.metadata or {}

        file_name = str(meta.get("file_name", "")).lower()
        page = meta.get("page")
        table_number = str(meta.get("table_number", ""))

        file_ok = True
        page_ok = True
        table_ok = True

        if requested_file:
            file_ok = requested_file in file_name or file_name in requested_file

        if requested_page is not None:
            try:
                page_ok = int(page) == requested_page
            except Exception:
                page_ok = False

        if requested_table:
            table_ok = table_number == requested_table

        if file_ok and page_ok and table_ok:
            matched.append(doc)

    return matched


def custom_keyword_score(question: str, doc_text: str, metadata: dict) -> float:
    """
    Custom score to boost exact table/page/file and keyword matches.

    This helps table questions because semantic embeddings alone can miss
    exact table rows or exact policy terms.
    """
    q_norm = normalize_text(question)
    d_norm = normalize_text(doc_text)

    q_tokens = tokenize(question)
    d_token_set = set(tokenize(doc_text))

    score = 0.0

    file_name = normalize_text(metadata.get("file_name", ""))
    page = str(metadata.get("page", ""))
    table_number = str(metadata.get("table_number", ""))
    source_kind = metadata.get("source_kind", "")

    # File boost
    if file_name and file_name in q_norm:
        score += 80.0

    # Page boost
    if page and f"page {page}" in q_norm:
        score += 60.0

    # Table boost
    if table_number and f"table {table_number}" in q_norm:
        score += 90.0

    # Table-specific boost
    if source_kind in {"table", "table_row"}:
        table_words = [
            "table",
            "row",
            "column",
            "percent",
            "number",
            "indicator",
            "policy",
            "regulatory",
            "incentive",
            "management",
            "financing",
            "monitoring",
            "land-use",
            "land",
            "burden",
            "requirements",
            "conditionality",
            "compliance",
        ]

        if any(word in q_norm for word in table_words):
            score += 35.0

    # Token coverage
    matched = 0

    for token in q_tokens:
        if token in d_token_set or token in d_norm:
            matched += 1
            score += 3.5

    if q_tokens:
        coverage = matched / len(q_tokens)
        score += coverage * 45.0

    # Phrase boost
    words = q_norm.split()

    for n in [7, 6, 5, 4, 3]:
        for i in range(0, max(0, len(words) - n + 1)):
            phrase = " ".join(words[i:i + n])

            if phrase and phrase in d_norm:
                score += 18.0 + n

    return score


def keyword_search_documents(vectorstore, question: str, top_k: int = 12):
    """
    Search all docs using custom keyword scoring.
    """
    all_docs = get_all_vector_docs(vectorstore)
    scored = []

    for doc in all_docs:
        meta = doc.metadata or {}
        score = custom_keyword_score(question, doc.page_content or "", meta)

        if score > 0:
            scored.append((score, doc))

    scored.sort(key=lambda x: x[0], reverse=True)

    debug = []

    for score, doc in scored[:5]:
        meta = doc.metadata or {}

        debug.append({
            "score": round(score, 2),
            "file_name": meta.get("file_name"),
            "page": meta.get("page"),
            "kind": meta.get("source_kind"),
            "table": meta.get("table_number"),
            "row": meta.get("row_index"),
        })

    return [doc for _, doc in scored[:top_k]], {
        "keyword_used": True,
        "top_keyword_score": scored[0][0] if scored else 0.0,
        "top_keyword_matches": debug,
    }


def bm25_search_documents(vectorstore, question: str, top_k: int = 12):
    """
    BM25 search if rank_bm25 is available.
    If rank_bm25 is unavailable, use a simple BM25-like fallback.
    """
    all_docs = get_all_vector_docs(vectorstore)

    if not all_docs:
        return [], {
            "bm25_available": RANK_BM25_AVAILABLE,
            "bm25_used": False,
        }

    tokenized_docs = [tokenize(doc.page_content or "") for doc in all_docs]
    query_tokens = tokenize(question)

    if not query_tokens:
        return [], {
            "bm25_available": RANK_BM25_AVAILABLE,
            "bm25_used": False,
        }

    if RANK_BM25_AVAILABLE:
        bm25 = BM25Okapi(tokenized_docs)
        scores = bm25.get_scores(query_tokens)

        scored = []

        for score, doc in zip(scores, all_docs):
            if score > 0:
                scored.append((float(score), doc))

        scored.sort(key=lambda x: x[0], reverse=True)

        return [doc for _, doc in scored[:top_k]], {
            "bm25_available": True,
            "bm25_used": True,
            "top_bm25_score": scored[0][0] if scored else 0.0,
        }

    # Fallback if rank_bm25 package is unavailable
    N = len(all_docs)

    df = {}

    for tokens in tokenized_docs:
        for token in set(tokens):
            df[token] = df.get(token, 0) + 1

    scored = []

    for doc, tokens in zip(all_docs, tokenized_docs):
        tf = {}

        for token in tokens:
            tf[token] = tf.get(token, 0) + 1

        score = 0.0
        doc_len = max(len(tokens), 1)

        for query_token in query_tokens:
            if query_token not in tf:
                continue

            idf = math.log((N + 1) / (df.get(query_token, 0) + 1)) + 1
            score += idf * (tf[query_token] / doc_len) * 100

        if score > 0:
            scored.append((score, doc))

    scored.sort(key=lambda x: x[0], reverse=True)

    return [doc for _, doc in scored[:top_k]], {
        "bm25_available": False,
        "bm25_used": True,
        "top_bm25_score": scored[0][0] if scored else 0.0,
    }


def dedupe_docs(docs):
    """
    Remove duplicate retrieved docs.
    """
    seen = set()
    output = []

    for doc in docs:
        meta = doc.metadata or {}

        key = (
            meta.get("doc_id"),
            meta.get("file_name"),
            meta.get("page"),
            meta.get("source_kind"),
            meta.get("table_number"),
            meta.get("row_index"),
            (doc.page_content or "")[:100],
        )

        if key in seen:
            continue

        seen.add(key)
        output.append(doc)

    return output