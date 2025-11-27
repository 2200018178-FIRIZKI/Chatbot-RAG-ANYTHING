import os
import warnings
from dotenv import load_dotenv

# --------------------------------------------------------
# Load environment variables from .env in the project root
# --------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(BASE_DIR, ".env")
load_dotenv(ENV_PATH)

# --------------------------------------------------------
# API Configuration with Validation
# --------------------------------------------------------

# Support both OPENROUTER_API_KEY and OPENAI_API_KEY for flexibility
OPENAI_API_KEY = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")  # Default OpenRouter

# Validate API key
if not OPENAI_API_KEY:
    warnings.warn(
        "⚠️  API key tidak ditemukan! Set OPENROUTER_API_KEY atau OPENAI_API_KEY di .env file",
        UserWarning
    )

# --------------------------------------------------------
# Parsing & RAG Configuration
# --------------------------------------------------------

OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./output")

PARSER = os.getenv("PARSER", "mineru")
PARSE_METHOD = os.getenv("PARSE_METHOD", "ocr")

# --------------------------------------------------------
# LLM Configuration
# --------------------------------------------------------

LLM_MODEL = os.getenv("LLM_MODEL", "meta-llama/llama-3.1-8b-instruct:free")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "2048"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))

# --------------------------------------------------------
# Embedding Configuration  
# --------------------------------------------------------

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "384"))

# --------------------------------------------------------
# Query Configuration
# --------------------------------------------------------

DEFAULT_QUERY_MODE = os.getenv("QUERY_MODE", "hybrid")
TOP_K_ENTITIES = int(os.getenv("TOP_K_ENTITIES", "40"))
COSINE_THRESHOLD = float(os.getenv("COSINE_THRESHOLD", "0.2"))

# --------------------------------------------------------
# Multimodal OFF (Untuk efisiensi dan penghematan token)
# --------------------------------------------------------

MULTIMODAL = False
MULTIMODAL_IMAGE = False
MULTIMODAL_TABLE = False
MULTIMODAL_EQUATION = False

# --------------------------------------------------------
# RAG Storage Configuration with Error Handling
# --------------------------------------------------------

WORKING_DIR = os.path.join(BASE_DIR, "rag_storage")

# Create directories with error handling
try:
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
except PermissionError as e:
    raise PermissionError(f"❌ Tidak bisa membuat direktori: {e}")
except Exception as e:
    raise Exception(f"❌ Error membuat direktori: {e}")

# --------------------------------------------------------
# Validation Summary
# --------------------------------------------------------

def validate_config():
    """Validate all configuration settings"""
    errors = []
    
    if not OPENAI_API_KEY:
        errors.append("API key tidak tersedia")
    
    if not os.path.exists(WORKING_DIR):
        errors.append(f"WORKING_DIR tidak dapat diakses: {WORKING_DIR}")
        
    if not os.path.exists(OUTPUT_DIR):
        errors.append(f"OUTPUT_DIR tidak dapat diakses: {OUTPUT_DIR}")
        
    if errors:
        raise ValueError(f"❌ Konfigurasi tidak valid: {'; '.join(errors)}")
    
    return True

# Auto-validate on import (optional, can be commented out)
# validate_config()
