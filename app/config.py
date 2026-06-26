import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
VECTOR_DIR = BASE_DIR / "vectorstore"
FAISS_INDEX_PATH = VECTOR_DIR / "faiss.index"
METADATA_PATH = VECTOR_DIR / "metadata.pkl"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "whisper-1")

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K_DENSE = 6
TOP_K_SPARSE = 6
TOP_K_FINAL = 5

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_DIR.mkdir(parents=True, exist_ok=True)
