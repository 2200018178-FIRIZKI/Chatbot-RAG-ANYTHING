import os
from dotenv import load_dotenv

# --------------------------------------------------------
# Load environment variables from .env in the project root
# --------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(BASE_DIR, ".env")
load_dotenv(ENV_PATH)

# --------------------------------------------------------
# API Configuration
# --------------------------------------------------------

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

# --------------------------------------------------------
# Parsing & RAG Configuration
# --------------------------------------------------------

OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./output")

PARSER = os.getenv("PARSER", "mineru")
PARSE_METHOD = os.getenv("PARSE_METHOD", "ocr")

# --------------------------------------------------------
# Multimodal OFF (WAJIB untuk OpenRouter FREE)
# --------------------------------------------------------

MULTIMODAL = False
MULTIMODAL_IMAGE = False
MULTIMODAL_TABLE = False
MULTIMODAL_EQUATION = False

# --------------------------------------------------------
# RAG Storage Configuration
# --------------------------------------------------------

WORKING_DIR = os.path.join(BASE_DIR, "rag_storage")

os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
