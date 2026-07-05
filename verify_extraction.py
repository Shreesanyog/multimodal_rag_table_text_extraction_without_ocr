import json
import sqlite3
from pathlib import Path

from config import PARSED_CACHE_DIR, EXTRACTION_REPORT_DIR, DB_PATH


OUTPUT_DIR = Path("verification_output")
OUTPUT_DIR.mkdir(exist_ok=True)

TABLES_DIR = OUTPUT_DIR / "structured_tables"
TEXT_DIR = OUTPUT_DIR / "page_text_samples"
TABLES_DIR.mkdir(exist_ok=True)
TEXT_DIR.mkdir(exist_ok=True)


def safe_name(text: str) -> str:
    text = str(text or "")
    for ch in ['\\', '/', ':', '*', '?', '"', '<', '>', '|']:
        text = text.replace(ch, "_")
    return text[:160]


def inspect_sqlite_documents():
    print("\n==============================")
    print("SQLITE DOCUMENT METADATA")
    print("==============================")

    if not DB_PATH.exists():
        print(f"DB not found: {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("""
        SELECT doc_id, file_name, status, pages, chunks, structured_tables, ocr_future_items
        FROM documents
        ORDER BY file_name
    """)

    rows = c.fetchall()
    conn.close()

    for row in rows:
        doc_id, file_name, status, pages, chunks, tables, future = row
        print(f"\nFile: {file_name}")
        print(f"Doc ID: {doc_id}")
        print(f"Status: {status}")
        print(f"Indexed pages: {pages}")
        print(f"Chunks: {chunks}")
        print(f"Structured tables: {tables}")
        print(f"Future OCR items: {future}")


def inspect_parsed_cache():
    print("\n==============================")
    print("PARSED CACHE INSPECTION")
    print("==============================")

    cache_files = sorted(PARSED_CACHE_DIR.glob("*_text_table_only.json"))

    if not cache_files:
        print(f"No parsed cache found in: {PARSED_CACHE_DIR}")
        return

    summary_lines = []

    for cache_file in cache_files:
        print(f"\nReading: {cache_file.name}")

        with open(cache_file, "r", encoding="utf-8") as f:
            payload = json.load(f)

        documents = payload.get("documents", [])
        report = payload.get("extraction_report", {})

        file_name = report.get("file_name", cache_file.name)
        doc_id = report.get("doc_id", "")
        structured_tables = report.get("structured_tables", [])
        ocr_future_items = report.get("ocr_future_items", [])

        text_docs = []
        full_table_docs = []
        table_row_docs = []

        for item in documents:
            meta = item.get("metadata", {})
            kind = meta.get("source_kind")
            chunk_type = meta.get("chunk_type")

            if kind == "text":
                text_docs.append(item)
            elif kind == "table" or chunk_type == "full_table":
                full_table_docs.append(item)
            elif kind == "table_row" or chunk_type == "table_row":
                table_row_docs.append(item)

        print(f"File: {file_name}")
        print(f"Doc ID: {doc_id}")
        print(f"Total document objects: {len(documents)}")
        print(f"Text page docs: {len(text_docs)}")
        print(f"Full table docs: {len(full_table_docs)}")
        print(f"Table row docs: {len(table_row_docs)}")
        print(f"Structured table summary count: {len(structured_tables)}")
        print(f"OCR future items: {len(ocr_future_items)}")

        summary_lines.append("=" * 100)
        summary_lines.append(f"FILE: {file_name}")
        summary_lines.append(f"DOC ID: {doc_id}")
        summary_lines.append(f"TOTAL DOCUMENT OBJECTS: {len(documents)}")
        summary_lines.append(f"TEXT PAGE DOCS: {len(text_docs)}")
        summary_lines.append(f"FULL TABLE DOCS: {len(full_table_docs)}")
        summary_lines.append(f"TABLE ROW DOCS: {len(table_row_docs)}")
        summary_lines.append(f"STRUCTURED TABLES: {len(structured_tables)}")
        summary_lines.append(f"OCR FUTURE ITEMS: {len(ocr_future_items)}")
        summary_lines.append("")

        summary_lines.append("STRUCTURED TABLES FOUND:")
        for t in structured_tables:
            summary_lines.append(
                f"- Table {t.get('table_number')} | Page {t.get('page')} | "
                f"Rows {t.get('rows')} | Columns {t.get('columns')} | Method {t.get('method')}"
            )
            summary_lines.append(f"  Caption: {t.get('caption')}")

        summary_lines.append("")
        summary_lines.append("OCR/FUTURE ITEMS:")
        for item in ocr_future_items:
            summary_lines.append(
                f"- {item.get('item_type')} {item.get('number', '')} | "
                f"Page {item.get('page')} | Reason: {item.get('reason')}"
            )
            if item.get("caption"):
                summary_lines.append(f"  Caption: {item.get('caption')}")

        summary_lines.append("")

        # Export full structured table chunks as markdown/txt
        for table_doc in full_table_docs:
            meta = table_doc.get("metadata", {})
            table_number = meta.get("table_number", "unknown")
            page = meta.get("page", "unknown")
            caption = meta.get("caption", "")

            out_name = safe_name(f"{file_name}_page_{page}_table_{table_number}.md")
            out_path = TABLES_DIR / out_name

            with open(out_path, "w", encoding="utf-8") as f:
                f.write(f"# {file_name} | Page {page} | Table {table_number}\n\n")
                f.write(f"Caption: {caption}\n\n")
                f.write("## Metadata\n\n")
                f.write(json.dumps(meta, indent=2, ensure_ascii=False))
                f.write("\n\n## Extracted table content\n\n")
                f.write(table_doc.get("page_content", ""))

            print(f"Exported table: {out_path}")

        # Export first few page text docs for quick checking
        for text_doc in text_docs[:5]:
            meta = text_doc.get("metadata", {})
            page = meta.get("page", "unknown")

            out_name = safe_name(f"{file_name}_page_{page}_text_sample.txt")
            out_path = TEXT_DIR / out_name

            with open(out_path, "w", encoding="utf-8") as f:
                f.write(f"FILE: {file_name}\n")
                f.write(f"PAGE: {page}\n")
                f.write("METADATA:\n")
                f.write(json.dumps(meta, indent=2, ensure_ascii=False))
                f.write("\n\nCONTENT:\n")
                f.write(text_doc.get("page_content", "")[:5000])

    summary_path = OUTPUT_DIR / "extraction_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    print("\n==============================")
    print("VERIFICATION EXPORT DONE")
    print("==============================")
    print(f"Summary: {summary_path}")
    print(f"Structured tables: {TABLES_DIR}")
    print(f"Text samples: {TEXT_DIR}")


def inspect_ocr_future_manifests():
    print("\n==============================")
    print("OCR FUTURE MANIFEST FILES")
    print("==============================")

    files = sorted(EXTRACTION_REPORT_DIR.glob("*_ocr_future_manifest.json"))

    if not files:
        print(f"No OCR future manifests found in: {EXTRACTION_REPORT_DIR}")
        return

    for file in files:
        with open(file, "r", encoding="utf-8") as f:
            payload = json.load(f)

        print(f"\nManifest: {file.name}")
        print(f"File: {payload.get('file_name')}")
        print(f"Structured tables: {len(payload.get('structured_tables', []))}")
        print(f"OCR future items: {len(payload.get('ocr_future_items', []))}")


def main():
    inspect_sqlite_documents()
    inspect_parsed_cache()
    inspect_ocr_future_manifests()


if __name__ == "__main__":
    main()