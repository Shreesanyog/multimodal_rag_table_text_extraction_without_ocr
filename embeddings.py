from pathlib import Path
from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

from config import LOCAL_HF_MODEL_PATH



DEFAULT_LOCAL_HF_PATH = r"D:\c9745ed1d9f207416be6d2e6f8de32d1f16199bf 2\c9745ed1d9f207416be6d2e6f8de32d1f16199bf"


@lru_cache(maxsize=1)
def get_embeddings():
    """
    Local HuggingFace embeddings only.

    Priority:
    1. LOCAL_HF_MODEL_PATH from .env / config.py
    2. DEFAULT_LOCAL_HF_PATH from your old working code

    No Gemini embedding fallback.
    """

    candidate_paths = [
        LOCAL_HF_MODEL_PATH,
        DEFAULT_LOCAL_HF_PATH,
    ]

    checked_paths = []

    for path in candidate_paths:
        if not path:
            continue

        model_path = Path(path)
        checked_paths.append(str(model_path))

        if model_path.exists():
            print(f"Using local HuggingFace embeddings from: {model_path}")

            return HuggingFaceEmbeddings(
                model_name=str(model_path),
                model_kwargs={"local_files_only": True},
                encode_kwargs={"normalize_embeddings": True},
            )

    raise FileNotFoundError(
        "Local HuggingFace model path not found.\n\n"
        "Checked these paths:\n"
        + "\n".join(f"- {p}" for p in checked_paths)
        + "\n\nFix options:\n"
        "1. Put the model folder at one of the checked paths, OR\n"
        "2. Set LOCAL_HF_MODEL_PATH correctly in .env.\n"
        "No Gemini embedding fallback is enabled in this version."
    )