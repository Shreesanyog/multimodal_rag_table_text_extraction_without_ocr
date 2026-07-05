from database import mark_document_deleted, rename_document
from ingest import load_vectorstore, save_vectorstore


def delete_document_from_faiss(doc_id: str) -> bool:
    vectorstore = load_vectorstore()

    if vectorstore is None:
        return False

    ids_to_delete = []

    try:
        for vector_id, docstore_id in vectorstore.index_to_docstore_id.items():
            doc = vectorstore.docstore.search(docstore_id)

            if not doc:
                continue

            meta = doc.metadata or {}

            if meta.get("doc_id") == doc_id:
                ids_to_delete.append(meta.get("chunk_id") or docstore_id)

    except Exception as e:
        print(f"Collect delete ids failed: {e}")
        return False

    if not ids_to_delete:
        mark_document_deleted(doc_id)
        return True

    try:
        vectorstore.delete(ids=ids_to_delete)
        save_vectorstore(vectorstore)
        mark_document_deleted(doc_id)
        return True
    except Exception as e:
        print(f"FAISS delete failed: {e}")
        return False


def rename_kb_document(doc_id: str, display_name: str) -> bool:
    return rename_document(doc_id, display_name)