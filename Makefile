# =============================================================================
# RAG-Anything Chatbot - Makefile
# =============================================================================
# Structured commands for running RAG chatbot system in organized sequence
#
# Usage:
#   make setup    - Initial environment setup
#   make install  - Install dependencies  
#   make process  - Process PDF documents
#   make chat     - Start interactive chat
#   make test     - Run system tests
#   make clean    - Clean temporary files
# =============================================================================

# Variables
PYTHON := python
VENV_NAME := venv
VENV_PATH := ./$(VENV_NAME)
PYTHON_VENV := $(VENV_PATH)/bin/python
PIP_VENV := $(VENV_PATH)/bin/pip
DATA_DIR := ./data
RAG_STORAGE := ./rag_storage
OUTPUT_DIR := ./output

# Colors for output
YELLOW := \033[1;33m
GREEN := \033[1;32m
RED := \033[1;31m
BLUE := \033[1;34m
RESET := \033[0m

# Default target
.PHONY: help
help:
	@echo "$(YELLOW)RAG-Anything Chatbot - Available Commands:$(RESET)"
	@echo ""
	@echo "$(GREEN)🚀 SETUP & INSTALLATION:$(RESET)"
	@echo "  make setup      - Create virtual environment"
	@echo "  make install    - Install Python dependencies"
	@echo "  make fix-mineru - Fix MinU parser installation"
	@echo "  make config     - Setup API configuration"
	@echo "  make init       - Complete initialization (setup + install + config)"
	@echo ""
	@echo "$(BLUE)📄 DATA PROCESSING:$(RESET)"
	@echo "  make process         - Process PDF documents into RAG storage"
	@echo "  make reprocess       - Clean storage and reprocess all documents"
	@echo "  make rebuild-embeddings - Rebuild with Hugging Face embeddings"
	@echo "  make status          - Show processing status and storage info"
	@echo ""
	@echo "$(GREEN)🤖 CHAT & QUERY:$(RESET)"
	@echo "  make chat      - Start interactive chatbot"
	@echo "  make query Q=\"your question\" - Ask single question"
	@echo "  make demo      - Run demo queries"
	@echo ""
	@echo "$(YELLOW)🔧 DEVELOPMENT & TESTING:$(RESET)"
	@echo "  make test              - Run quick system tests"
	@echo "  make test-quick        - Quick functionality test"
	@echo "  make test-comprehensive - Full comprehensive testing"
	@echo "  make test-compatible   - Test compatible RAG system (WORKING!)"
	@echo "  make test-interactive  - Interactive demo testing"
	@echo "  make test-all          - Run all tests"
	@echo "  make health-check      - System health check"
	@echo "  make status-check      - System status check"
	@echo "  make debug             - Debug RAG system status"
	@echo "  make lint              - Run code linting"
	@echo ""
	@echo "$(BLUE)🔧 COMPATIBILITY & FIXES:$(RESET)"
	@echo "  make fix-compatibility - Fix RAG system compatibility issues"
	@echo "  make load-data-compatible - Load data into compatible system"
	@echo ""
	@echo "$(RED)🧹 CLEANUP & MAINTENANCE:$(RESET)"
	@echo "  make clean     - Clean temporary files and cache"
	@echo "  make reset     - Complete reset (clean + remove storage)"
	@echo "  make logs      - Show recent logs"
	@echo ""

# =============================================================================
# 🚀 SETUP & INSTALLATION
# =============================================================================

.PHONY: setup
setup:
	@echo "$(YELLOW)🚀 Setting up virtual environment...$(RESET)"
	@if [ ! -d "$(VENV_PATH)" ]; then \
		$(PYTHON) -m venv $(VENV_NAME); \
		echo "$(GREEN)✅ Virtual environment created$(RESET)"; \
	else \
		echo "$(GREEN)✅ Virtual environment already exists$(RESET)"; \
	fi

.PHONY: install
install: setup
	@echo "$(YELLOW)📦 Installing dependencies...$(RESET)"
	@$(PIP_VENV) install --upgrade pip
	@$(PIP_VENV) install -r requirements.txt
	@echo "$(GREEN)✅ Dependencies installed$(RESET)"

.PHONY: install-mineru
install-mineru: setup
	@echo "$(YELLOW)⚙️ Installing MinU parser...$(RESET)"
	@$(PIP_VENV) install -U 'mineru[core]'
	@echo "$(GREEN)✅ MinU parser installed$(RESET)"

.PHONY: fix-mineru
fix-mineru: install-mineru
	@echo "$(YELLOW)🔧 Fixing MinU PATH issues...$(RESET)"
	@if [ ! -f "$(VENV_PATH)/bin/mineru" ]; then \
		echo "$(RED)❌ MinU not found in venv$(RESET)"; \
		$(PIP_VENV) install -U 'mineru[core]'; \
	else \
		echo "$(GREEN)✅ MinU found at $(VENV_PATH)/bin/mineru$(RESET)"; \
	fi
	@mkdir -p ~/.local/bin
	@ln -sf $(PWD)/$(VENV_PATH)/bin/mineru ~/.local/bin/mineru
	@echo "$(GREEN)✅ MinU symlink created in ~/.local/bin$(RESET)"
	@echo "$(BLUE)💡 Add ~/.local/bin to your PATH if needed$(RESET)"

.PHONY: config
config:
	@echo "$(YELLOW)⚙️ Setting up API configuration...$(RESET)"
	@if [ ! -f ".env" ]; then \
		echo "$(RED)❌ .env file not found!$(RESET)"; \
		echo "$(YELLOW)Please create .env file with your API keys$(RESET)"; \
		echo "$(BLUE)Example:$(RESET)"; \
		echo "OPENROUTER_API_KEY=your_key_here"; \
		echo "OPENAI_API_KEY=your_key_here"; \
		exit 1; \
	else \
		echo "$(GREEN)✅ .env file found$(RESET)"; \
	fi
	@$(PYTHON_VENV) test_api.py

.PHONY: init
init: setup install fix-mineru config
	@echo "$(GREEN)🎉 Complete initialization finished!$(RESET)"

# =============================================================================
# 📄 DATA PROCESSING
# =============================================================================

.PHONY: process
process:
	@echo "$(YELLOW)📄 Processing PDF documents...$(RESET)"
	@if [ ! -d "$(VENV_PATH)" ]; then \
		echo "$(RED)❌ Virtual environment not found. Run 'make setup' first$(RESET)"; \
		exit 1; \
	fi
	@if [ ! -f ".env" ]; then \
		echo "$(RED)❌ .env file not found. Run 'make config' first$(RESET)"; \
		exit 1; \
	fi
	@export PATH="$$HOME/.local/bin:$(PWD)/$(VENV_PATH)/bin:$$PATH" && $(PYTHON_VENV) -c "from app.process_pdf import run_process_pdf; run_process_pdf()"
	@echo "$(GREEN)✅ PDF processing completed$(RESET)"

.PHONY: reprocess
reprocess: clean-storage process
	@echo "$(GREEN)✅ Complete reprocessing finished$(RESET)"

.PHONY: rebuild-embeddings
rebuild-embeddings: clean-storage
	@echo "$(YELLOW)🔄 Rebuilding with Hugging Face embeddings...$(RESET)"
	@export PATH="$$HOME/.local/bin:$(PWD)/$(VENV_PATH)/bin:$$PATH" && $(PYTHON_VENV) -c "from app.process_pdf import run_process_pdf; run_process_pdf()"
	@echo "$(GREEN)✅ Embeddings rebuilt with Hugging Face model$(RESET)"

.PHONY: status
status:
	@echo "$(YELLOW)📊 RAG System Status:$(RESET)"
	@echo ""
	@if [ -d "$(DATA_DIR)" ]; then \
		echo "$(BLUE)📄 PDF Files:$(RESET)"; \
		find $(DATA_DIR) -name "*.pdf" -exec basename {} \; 2>/dev/null | head -5 || echo "  No PDF files found"; \
		echo ""; \
	fi
	@if [ -d "$(RAG_STORAGE)" ]; then \
		echo "$(BLUE)🗄️ RAG Storage:$(RESET)"; \
		ls -la $(RAG_STORAGE) | tail -n +2 | wc -l | xargs echo "  Files count:"; \
		du -sh $(RAG_STORAGE) 2>/dev/null | cut -f1 | xargs echo "  Storage size:" || echo "  Storage size: 0"; \
		echo ""; \
	fi
	@if [ -d "$(OUTPUT_DIR)" ]; then \
		echo "$(BLUE)📤 Output Files:$(RESET)"; \
		find $(OUTPUT_DIR) -type f | wc -l | xargs echo "  Output files:"; \
		echo ""; \
	fi

# =============================================================================
# 🤖 CHAT & QUERY
# =============================================================================

.PHONY: chat
chat:
	@echo "$(YELLOW)🤖 Starting interactive chatbot...$(RESET)"
	@echo "$(BLUE)Type 'quit' or 'exit' to stop$(RESET)"
	@echo ""
	@export PATH="$$HOME/.local/bin:$(PWD)/$(VENV_PATH)/bin:$$PATH" && $(PYTHON_VENV) main.py

.PHONY: query
query:
	@if [ -z "$(Q)" ]; then \
		echo "$(RED)❌ Please provide a question: make query Q=\"your question\"$(RESET)"; \
		exit 1; \
	fi
	@echo "$(YELLOW)🔍 Query: $(Q)$(RESET)"
	@echo ""
	@$(PYTHON_VENV) -c "from app.query_chatbot import ask; print(ask('$(Q)'))"

.PHONY: demo
demo:
	@echo "$(YELLOW)🎯 Running demo queries...$(RESET)"
	@echo ""
	@echo "$(BLUE)Query 1: Apa isi utama dokumen ini?$(RESET)"
	@$(PYTHON_VENV) -c "from app.query_chatbot import ask; print(ask('Apa isi utama dokumen ini?'))"
	@echo ""
	@echo "$(BLUE)Query 2: Berapa total anggaran yang ada?$(RESET)"
	@$(PYTHON_VENV) -c "from app.query_chatbot import ask; print(ask('Berapa total anggaran yang ada?'))"
	@echo ""
	@echo "$(GREEN)✅ Demo completed$(RESET)"

# =============================================================================
# 🔧 DEVELOPMENT & TESTING
# =============================================================================

.PHONY: test
test:
	@echo "$(YELLOW)🧪 Running system tests...$(RESET)"
	@echo ""
	@echo "$(BLUE)1. Testing API connection...$(RESET)"
	@$(PYTHON_VENV) test_api.py
	@echo ""
	@echo "$(BLUE)2. Testing RAG initialization...$(RESET)"
	@$(PYTHON_VENV) -c "from app.rag_init import get_rag; rag = get_rag(); print('✅ RAG instance created successfully')"
	@echo ""
	@echo "$(BLUE)3. Testing query system...$(RESET)"
	@$(PYTHON_VENV) -c "from app.query_chatbot import ask; result = ask('test'); print('✅ Query system working')"
	@echo ""
	@echo "$(GREEN)✅ All tests completed$(RESET)"

.PHONY: debug
debug:
	@echo "$(YELLOW)🔍 RAG System Debug Information:$(RESET)"
	@echo ""
	@$(PYTHON_VENV) -c "\
import os; \
from app.rag_init import get_rag; \
print('📁 Working Directory:', os.getcwd()); \
print('🔑 Environment Variables:'); \
print('  OPENROUTER_API_KEY:', 'SET' if os.getenv('OPENROUTER_API_KEY') else 'NOT SET'); \
print('  OPENAI_API_KEY:', 'SET' if os.getenv('OPENAI_API_KEY') else 'NOT SET'); \
print(''); \
rag = get_rag(); \
print('🤖 RAG Instance:', type(rag).__name__); \
print('💾 LightRAG:', 'INITIALIZED' if hasattr(rag, 'lightrag') and rag.lightrag else 'NOT INITIALIZED'); \
"

.PHONY: lint
lint:
	@echo "$(YELLOW)🔍 Running code linting...$(RESET)"
	@if command -v flake8 >/dev/null 2>&1; then \
		$(PYTHON_VENV) -m flake8 app/ --max-line-length=100 --ignore=E501,W503; \
	else \
		echo "$(BLUE)flake8 not installed, skipping lint$(RESET)"; \
	fi

# =============================================================================
# 🧪 TESTING & QUALITY ASSURANCE
# =============================================================================

.PHONY: test-quick
test-quick:
	@echo "$(GREEN)🚀 Quick Chatbot Test$(RESET)"
	@$(PYTHON_VENV) test_chatbot_quick.py

.PHONY: test-comprehensive
test-comprehensive:
	@echo "$(GREEN)🧪 Comprehensive Chatbot Testing$(RESET)"
	@$(PYTHON_VENV) test_chatbot_comprehensive.py

.PHONY: test-interactive
test-interactive:
	@echo "$(GREEN)🎮 Interactive Demo Testing$(RESET)"
	@echo "$(YELLOW)⚠️  Make sure chatbot is working before running interactive test$(RESET)"
	@$(PYTHON_VENV) -c "from test_chatbot_quick import test_interactive_demo; test_interactive_demo()"

.PHONY: test-all
test-all: test-quick test-comprehensive
	@echo "$(GREEN)✅ All tests completed$(RESET)"

.PHONY: test-compatible
test-compatible:
	@echo "$(GREEN)🤖 Testing Compatible RAG System$(RESET)"
	@$(PYTHON_VENV) test_compatible.py

.PHONY: load-data-compatible
load-data-compatible:
	@echo "$(GREEN)📚 Loading Data into Compatible System$(RESET)"
	@$(PYTHON_VENV) load_data_compatible.py

.PHONY: fix-compatibility
fix-compatibility:
	@echo "$(GREEN)🔧 Fixing RAG System Compatibility$(RESET)"
	@$(PYTHON_VENV) fix_rag_compatibility.py

.PHONY: health-check
health-check:
	@echo "$(GREEN)🏥 System Health Check$(RESET)"
	@$(PYTHON_VENV) -c "from app.rag_init_simple import health_check; print('Health:', health_check())"

.PHONY: status-check
status-check:
	@echo "$(GREEN)📊 System Status Check$(RESET)"
	@$(PYTHON_VENV) -c "from main import RAGChatbotApp; app = RAGChatbotApp(); app.initialize_system(); app.show_system_status()"

# =============================================================================
# 🧹 CLEANUP & MAINTENANCE
# =============================================================================

.PHONY: clean
clean:
	@echo "$(YELLOW)🧹 Cleaning temporary files...$(RESET)"
	@find . -type f -name "*.pyc" -delete
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.log" -delete 2>/dev/null || true
	@echo "$(GREEN)✅ Cleanup completed$(RESET)"

.PHONY: clean-storage
clean-storage:
	@echo "$(YELLOW)🗑️ Cleaning RAG storage...$(RESET)"
	@if [ -d "$(RAG_STORAGE)" ]; then \
		rm -rf $(RAG_STORAGE)/*; \
		echo "$(GREEN)✅ RAG storage cleaned$(RESET)"; \
	else \
		echo "$(BLUE)RAG storage directory not found$(RESET)"; \
	fi

.PHONY: reset
reset: clean clean-storage
	@echo "$(YELLOW)🔄 Complete system reset...$(RESET)"
	@if [ -d "$(OUTPUT_DIR)" ]; then \
		rm -rf $(OUTPUT_DIR)/*; \
		echo "$(GREEN)✅ Output directory cleaned$(RESET)"; \
	fi
	@echo "$(GREEN)✅ Complete reset finished$(RESET)"

.PHONY: logs
logs:
	@echo "$(YELLOW)📋 Recent system logs:$(RESET)"
	@if [ -f "app.log" ]; then \
		tail -20 app.log; \
	else \
		echo "$(BLUE)No log file found$(RESET)"; \
	fi

# =============================================================================
# 🔒 ENVIRONMENT CHECKS
# =============================================================================

.PHONY: check-env
check-env:
	@if [ ! -d "$(VENV_PATH)" ]; then \
		echo "$(RED)❌ Virtual environment not found$(RESET)"; \
		echo "$(YELLOW)Run: make setup$(RESET)"; \
		exit 1; \
	fi

.PHONY: check-config
check-config:
	@if [ ! -f ".env" ]; then \
		echo "$(RED)❌ Configuration file not found$(RESET)"; \
		echo "$(YELLOW)Run: make config$(RESET)"; \
		exit 1; \
	fi

# Dependencies for main targets
process: check-env check-config
chat: check-env check-config  
query: check-env check-config
test: check-env check-config
debug: check-env

# =============================================================================
# 📋 DEVELOPMENT HELPERS
# =============================================================================

.PHONY: install-dev
install-dev: install
	@echo "$(YELLOW)📦 Installing development dependencies...$(RESET)"
	@$(PIP_VENV) install flake8 black pytest
	@echo "$(GREEN)✅ Development dependencies installed$(RESET)"

.PHONY: format
format:
	@echo "$(YELLOW)🎨 Formatting code...$(RESET)"
	@if command -v black >/dev/null 2>&1; then \
		$(PYTHON_VENV) -m black app/ --line-length=100; \
		echo "$(GREEN)✅ Code formatted$(RESET)"; \
	else \
		echo "$(BLUE)black not installed, skipping format$(RESET)"; \
	fi

.PHONY: requirements
requirements:
	@echo "$(YELLOW)📋 Updating requirements.txt...$(RESET)"
	@$(PIP_VENV) freeze > requirements.txt
	@echo "$(GREEN)✅ requirements.txt updated$(RESET)"

# =============================================================================
# 📚 INFO & DOCUMENTATION
# =============================================================================

.PHONY: info
info:
	@echo "$(BLUE)📚 RAG-Anything Chatbot System$(RESET)"
	@echo ""
	@echo "$(YELLOW)Architecture:$(RESET)"
	@echo "  • RAGAnything framework for document processing"
	@echo "  • LightRAG for knowledge graph and vector storage"
	@echo "  • OpenRouter API with Meta Llama 3.1 8B model"
	@echo "  • MinIO parser for PDF OCR processing"
	@echo ""
	@echo "$(YELLOW)Quick Start:$(RESET)"
	@echo "  1. make init        # Setup environment"
	@echo "  2. make process     # Process your PDFs"
	@echo "  3. make chat        # Start chatting!"
	@echo ""

.PHONY: version
version:
	@echo "$(BLUE)📋 System Versions:$(RESET)"
	@$(PYTHON_VENV) --version
	@$(PIP_VENV) --version
	@$(PYTHON_VENV) -c "import raganything; print('RAGAnything:', raganything.__version__)" 2>/dev/null || echo "RAGAnything: Not installed"