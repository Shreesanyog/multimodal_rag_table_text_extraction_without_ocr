from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from config import PRELOADED_DIR, FAISS_DIR, TEXT_CHUNK_SIZE, TEXT_CHUNK_OVERLAP
from embeddings import get_embeddings
from document_parser import parse_pdf, get_extraction_report
from database import upsert_document, document_is_ready
from utils import stable_doc_id



# CHUNKING


def split_documents(docs: List[Document]) -> List[Document]:
    """
    Chunking strategy for text-table-only RAG.

    Normal PDF text:
    - Split into overlapping chunks.

    Tables:
    - Do not split.
    - Keep full table chunk.
    - Keep row-level chunks.
    """
    text_docs = []
    table_docs = []

    for doc in docs:
        metadata = doc.metadata or {}
        source_kind = metadata.get("source_kind", "")
        chunk_type = metadata.get("chunk_type", "")

        if source_kind in {"table", "table_row"} or chunk_type in {"full_table", "table_row"}:
            table_docs.append(doc)
        else:
            text_docs.append(doc)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=TEXT_CHUNK_SIZE,
        chunk_overlap=TEXT_CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""],
    )

    chunks = splitter.split_documents(text_docs)

    # Preserve table chunks as-is
    chunks.extend(table_docs)

    # Add stable chunk IDs
    for idx, chunk in enumerate(chunks):
        metadata = chunk.metadata or {}

        doc_id = metadata.get("doc_id", "unknown_doc")
        page = metadata.get("page", "unknown_page")
        source_kind = metadata.get("source_kind", "text")
        table_number = metadata.get("table_number")
        row_index = metadata.get("row_index")

        if table_number and row_index:
            chunk_id = f"{doc_id}_p{page}_{source_kind}_t{table_number}_r{row_index}_c{idx}"
        elif table_number:
            chunk_id = f"{doc_id}_p{page}_{source_kind}_t{table_number}_c{idx}"
        else:
            chunk_id = f"{doc_id}_p{page}_{source_kind}_c{idx}"

        chunk.metadata["chunk_index"] = idx
        chunk.metadata["chunk_id"] = chunk_id

    return chunks



# FAISS HELPERS


def index_exists() -> bool:
    return (FAISS_DIR / "index.faiss").exists() and (FAISS_DIR / "index.pkl").exists()


def load_vectorstore():
    if not index_exists():
        return None

    return FAISS.load_local(
        str(FAISS_DIR),
        get_embeddings(),
        allow_dangerous_deserialization=True,
    )


def save_vectorstore(vectorstore):
    FAISS_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(FAISS_DIR))



# SINGLE PDF INGESTION


def ingest_pdf(pdf_path: str, source_type: str = "uploaded", force: bool = False) -> int:
    pdf_path = str(pdf_path)
    path_obj = Path(pdf_path)

    if not path_obj.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc_id = stable_doc_id(pdf_path)
    file_name = path_obj.name

    if not force and index_exists() and document_is_ready(doc_id):
        print(f"Skipping already indexed document: {file_name}")
        return 0

    upsert_document(
        doc_id=doc_id,
        file_name=file_name,
        file_path=pdf_path,
        source_type=source_type,
        status="processing",
        pages=0,
        chunks=0,
        structured_tables=0,
        ocr_future_items=0,
    )

    try:
        docs = parse_pdf(pdf_path)
        chunks = split_documents(docs)

        if not chunks:
            upsert_document(
                doc_id=doc_id,
                file_name=file_name,
                file_path=pdf_path,
                source_type=source_type,
                status="failed",
                pages=0,
                chunks=0,
                structured_tables=0,
                ocr_future_items=0,
            )
            return 0

        embeddings = get_embeddings()
        existing_vectorstore = load_vectorstore()

        ids = [chunk.metadata["chunk_id"] for chunk in chunks]

        if existing_vectorstore is None:
            vectorstore = FAISS.from_documents(
                documents=chunks,
                embedding=embeddings,
                ids=ids,
            )
        else:
            existing_vectorstore.add_documents(
                documents=chunks,
                ids=ids,
            )
            vectorstore = existing_vectorstore

        save_vectorstore(vectorstore)

        report = get_extraction_report(pdf_path)

        structured_tables_count = len(report.get("structured_tables", []))
        ocr_future_items_count = len(report.get("ocr_future_items", []))
        pages_count = len({
            doc.metadata.get("page")
            for doc in docs
            if doc.metadata.get("page")
        })

        upsert_document(
            doc_id=doc_id,
            file_name=file_name,
            file_path=pdf_path,
            source_type=source_type,
            status="ready",
            pages=pages_count,
            chunks=len(chunks),
            structured_tables=structured_tables_count,
            ocr_future_items=ocr_future_items_count,
        )

        print(f"Indexed: {file_name}")
        print(f"Pages: {pages_count}")
        print(f"Chunks: {len(chunks)}")
        print(f"Structured tables: {structured_tables_count}")
        print(f"Future OCR items: {ocr_future_items_count}")

        return len(chunks)

    except Exception as e:
        upsert_document(
            doc_id=doc_id,
            file_name=file_name,
            file_path=pdf_path,
            source_type=source_type,
            status="failed",
            pages=0,
            chunks=0,
            structured_tables=0,
            ocr_future_items=0,
        )

        print(f"Ingestion failed for {file_name}: {e}")
        raise



# PRELOADED PDF INGESTION


def ingest_preloaded_pdfs(force: bool = False) -> int:
    total_chunks = 0
    pdfs = sorted(PRELOADED_DIR.glob("*.pdf"))

    if not pdfs:
        print(f"No PDFs found in {PRELOADED_DIR}")
        return 0

    print(f"Found {len(pdfs)} preloaded PDF(s).")

    for pdf in pdfs:
        print()
        print("----------------------------------------")
        print(f"Processing: {pdf.name}")
        print("----------------------------------------")

        added_chunks = ingest_pdf(
            pdf_path=str(pdf),
            source_type="preloaded",
            force=force,
        )

        total_chunks += added_chunks

    print()
    print("========================================")
    print(f"Total chunks added: {total_chunks}")
    print("========================================")

    return total_chunks