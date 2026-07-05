import hashlib
import re
from pathlib import Path


def stable_doc_id(file_path: str) -> str:
    p = Path(file_path)
    raw = f"{p.name}-{p.stat().st_size}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def clean_text(text: str) -> str:
    if not text:
        return ""

    text = text.replace("\x00", " ")
    text = text.replace("ﬁ", "fi").replace("ﬂ", "fl")
    text = text.replace("–", "-").replace("—", "-").replace("‑", "-")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_inline(text: str) -> str:
    return " ".join(str(text or "").replace("\n", " ").split())


def safe_filename(name: str) -> str:
    name = re.sub(r"[^a-zA-Z0-9_.-]+", "_", name or "file")
    return name[:180]


def table_to_markdown(rows):
    cleaned = []
    for row in rows:
        cleaned.append([normalize_inline(cell) for cell in row])

    cleaned = [r for r in cleaned if any(c.strip() for c in r)]
    if not cleaned:
        return ""

    max_cols = max(len(r) for r in cleaned)
    cleaned = [r + [""] * (max_cols - len(r)) for r in cleaned]

    header = cleaned[0]
    body = cleaned[1:]

    md = []
    md.append("| " + " | ".join(header) + " |")
    md.append("| " + " | ".join(["---"] * max_cols) + " |")

    for row in body:
        md.append("| " + " | ".join(row) + " |")

    return "\n".join(md)


def looks_like_contents_page(text: str) -> bool:
    upper = (text or "").upper()
    return "CONTENTS" in upper and "CORE MESSAGES" in upper


def should_skip_text_page(text: str) -> bool:
    upper = (text or "").upper()
    if len((text or "").split()) < 8:
        return True
    if looks_like_contents_page(text):
        return False
    return False