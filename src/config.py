"""
Config settings.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "documents"
VECTOR_STORE_PATH = BASE_DIR / "data" / "vector_store"

# Ollama
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")

# Embeddings
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

# RAG
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP"))
