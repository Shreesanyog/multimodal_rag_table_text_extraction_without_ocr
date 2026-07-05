import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
PRELOADED_DIR = DATA_DIR / "preloaded"
UPLOAD_DIR = DATA_DIR / "uploads"
STORAGE_DIR = BASE_DIR / "storage"
FAISS_DIR = STORAGE_DIR / "faiss_index"
PARSED_CACHE_DIR = STORAGE_DIR / "parsed_cache"
EXTRACTION_REPORT_DIR = STORAGE_DIR / "extraction_reports"
PAGE_IMAGES_DIR = STORAGE_DIR / "page_images"

DB_PATH = BASE_DIR / "app_data.db"

# Gemini only for final answer generation, not embeddings and not OCR
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GEMINI_CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-2.5-flash")

# Local HuggingFace embeddings only
LOCAL_HF_MODEL_PATH = os.getenv("LOCAL_HF_MODEL_PATH", "models/bge-small-en-v1.5")

# Login defaults
DEFAULT_ADMIN_USERNAME = os.getenv("DEFAULT_ADMIN_USERNAME", "admin")
DEFAULT_ADMIN_PASSWORD = os.getenv("DEFAULT_ADMIN_PASSWORD", "admin123")

# Chunking
TEXT_CHUNK_SIZE = 850
TEXT_CHUNK_OVERLAP = 140

# Retrieval
RETRIEVAL_K = 14
FINAL_CITATION_COUNT = 4

for folder in [
    DATA_DIR,
    PRELOADED_DIR,
    UPLOAD_DIR,
    STORAGE_DIR,
    FAISS_DIR,
    PARSED_CACHE_DIR,
    EXTRACTION_REPORT_DIR,
    PAGE_IMAGES_DIR,
]:
    folder.mkdir(parents=True, exist_ok=True)