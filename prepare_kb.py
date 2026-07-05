import shutil
from pathlib import Path

from config import FAISS_DIR, PARSED_CACHE_DIR, EXTRACTION_REPORT_DIR, DB_PATH
from database import init_db
from ingest import ingest_preloaded_pdfs


def remove_folder(path: Path):
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)


def main():
    print("============================================")
    print("Preparing text-table-only PDF knowledge base")
    print("============================================")
    print("Mode: extractable text + structured tables only")
    print("OCR/figures/flowcharts/color-coded visual items will be saved for future handling.")
    print()

    print("Cleaning old index/cache/reports/database...")
    remove_folder(FAISS_DIR)
    remove_folder(PARSED_CACHE_DIR)
    remove_folder(EXTRACTION_REPORT_DIR)

    if DB_PATH.exists():
        DB_PATH.unlink()

    init_db()

    print("Rebuilding all preloaded PDFs from scratch...")
    print()

    chunks = ingest_preloaded_pdfs(force=True)

    print()
    print("Knowledge base preparation completed.")
    print(f"Chunks added: {chunks}")
    print()
    print("Saved assets:")
    print(f"- FAISS index: {FAISS_DIR}")
    print(f"- Parsed cache: {PARSED_CACHE_DIR}")
    print(f"- OCR future manifests: {EXTRACTION_REPORT_DIR}")
    print("- SQLite metadata: app_data.db")


if __name__ == "__main__":
    main()