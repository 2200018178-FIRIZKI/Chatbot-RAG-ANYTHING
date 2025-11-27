import os
import logging
import asyncio
import concurrent.futures
from typing import List, Optional, Dict, Any
from raganything import RAGAnything, RAGAnythingConfig
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc

from .config import (
    OPENAI_API_KEY, OPENAI_BASE_URL, WORKING_DIR,
    LLM_MODEL, TEMPERATURE, MAX_TOKENS,
    EMBEDDING_MODEL, EMBEDDING_DIMENSION
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===================================================== 
# STORAGE & INSTANCE SINGLETON
# =====================================================
# Global instances
_rag_instance = None
_hf_model = None

def validate_configuration():
    """Validate required configuration before initialization"""
    if not OPENAI_API_KEY:
        raise ValueError("❌ OPENAI_API_KEY tidak tersedia. Set OPENROUTER_API_KEY di .env file")
    
    if not os.path.exists(WORKING_DIR):
        os.makedirs(WORKING_DIR, exist_ok=True)
        logger.info(f"✅ Created working directory: {WORKING_DIR}")
    
    return True

# =====================================================
# 1. LLM MODEL - STABLE VERSION FROM WORKING CODE
# =====================================================
def _create_llm_model_func():
    """Create LLM model function using proven working approach"""
    
    def llm_model_func(
        prompt: str, 
        system_prompt: Optional[str] = None, 
        history_messages: Optional[List] = None, 
        **kwargs
    ) -> str:
        if history_messages is None:
            history_messages = []
            
        try:
            # Use proven working parameters from original
            kwargs['max_tokens'] = MAX_TOKENS or 150
            kwargs['temperature'] = TEMPERATURE or 0.3
            
            response = openai_complete_if_cache(
                LLM_MODEL,  # Use model from config
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                api_key=OPENAI_API_KEY,
                base_url=OPENAI_BASE_URL,
                **kwargs
            )
            return response
            
        except Exception as e:
            logger.error(f"❌ LLM Error: {e}")
            return f"Error generating response: {str(e)[:100]}..."

    return llm_model_func

# =====================================================
# 2. EMBEDDING MODEL - STABLE ASYNC VERSION
# =====================================================
def _create_embedding_func():
    """Create async embedding function using proven working approach"""
    global _hf_model
    
    # Initialize model once - using working pattern from original
    if '_hf_model' not in globals():
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"🤖 Loading Hugging Face embedding model ({EMBEDDING_MODEL})...")
            _hf_model = SentenceTransformer(EMBEDDING_MODEL)
            logger.info("✅ Embedding model loaded successfully!")
        except Exception as e:
            logger.error(f"❌ Failed to load embedding model: {e}")
            _hf_model = None

    async def async_huggingface_embedding_func(texts):
        """Async wrapper using proven ThreadPoolExecutor approach"""
        if isinstance(texts, str):
            texts = [texts]
        
        try:
            if _hf_model is None:
                import numpy as np
                return [np.zeros(EMBEDDING_DIMENSION).tolist() for _ in texts]
            
            # Use proven working async pattern
            import asyncio
            import concurrent.futures
            
            def encode_texts(text_list):
                return _hf_model.encode(text_list)
            
            # Proven working ThreadPoolExecutor pattern
            with concurrent.futures.ThreadPoolExecutor() as executor:
                embeddings = await asyncio.get_event_loop().run_in_executor(
                    executor, encode_texts, texts
                )
            
            # Convert to list format expected by LightRAG
            return [emb.tolist() for emb in embeddings]
            
        except Exception as e:
            logger.error(f"❌ Hugging Face embedding error: {e}")
            # Return zero vectors as fallback
            import numpy as np
            return [np.zeros(EMBEDDING_DIMENSION).tolist() for _ in texts]

    return EmbeddingFunc(
        embedding_dim=EMBEDDING_DIMENSION,
        max_token_size=512,
        func=async_huggingface_embedding_func,
    )

# =====================================================
# 3. RAGAnything — IMPROVED SINGLETON WITH ERROR HANDLING
# =====================================================
# =====================================================
# 3. RAGAnything — ENHANCED SINGLETON WITH PROVEN PATTERN
# =====================================================
def get_rag() -> Optional[RAGAnything]:
    """Get RAG instance using proven working pattern with enhancements"""
    global _rag_instance
    
    if _rag_instance is not None:
        return _rag_instance

    try:
        # Validate configuration first
        validate_configuration()
        
        # Create LightRAG instance if storage exists - using proven pattern
        lightrag_instance = None
        if os.path.exists(os.path.join(WORKING_DIR, "kv_store_full_docs.json")):
            try:
                lightrag_instance = LightRAG(
                    working_dir=WORKING_DIR,
                    llm_model_func=_create_llm_model_func(),
                    embedding_func=_create_embedding_func(),
                    embedding_func_max_async=1,  # From proven working config
                    llm_model_max_async=1,       # From proven working config
                )
                logger.info("✅ LightRAG instance created from existing storage")
            except Exception as e:
                logger.warning(f"⚠️ Could not initialize LightRAG from storage: {e}")
                lightrag_instance = None
        else:
            logger.info("📁 No existing storage found. LightRAG will be created fresh.")
        
        # Create RAGAnything configuration
        config = RAGAnythingConfig(
            working_dir=WORKING_DIR,
            parser="mineru",
            parse_method="ocr",
            enable_image_processing=False,
            enable_table_processing=False,
            enable_equation_processing=False,
        )

        # Create function instances
        llm_model_func = _create_llm_model_func()
        embedding_func = _create_embedding_func()

        # Create RAGAnything instance with proven pattern
        _rag_instance = RAGAnything(
            config=config,
            llm_model_func=llm_model_func,
            embedding_func=embedding_func,
            lightrag=lightrag_instance  # Pass pre-initialized LightRAG
        )
        
        logger.info("✅ RAGAnything instance created successfully")
        return _rag_instance
        
    except Exception as e:
        logger.error(f"❌ Failed to create RAG instance: {e}")
        return None

def reset_rag_instance():
    """Reset RAG instance and clear model cache"""
    global _rag_instance
    
    logger.info("🔄 Resetting RAG instance...")
    _rag_instance = None
    
    # Optionally reset embedding model to free memory
    # global _hf_model
    # _hf_model = None  # Uncomment to free embedding model memory

def get_system_info() -> Dict[str, Any]:
    """Get comprehensive system information for debugging"""
    info = {
        "rag_instance_exists": _rag_instance is not None,
        "embedding_model_loaded": _hf_model is not None,
        "working_directory": WORKING_DIR,
        "working_dir_exists": os.path.exists(WORKING_DIR),
        "api_key_configured": bool(OPENAI_API_KEY),
        "config": {
            "llm_model": LLM_MODEL,
            "embedding_model": EMBEDDING_MODEL,
            "embedding_dimension": EMBEDDING_DIMENSION,
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS
        }
    }
    
    # Check storage files
    storage_files = [
        "kv_store_full_docs.json",
        "kv_store_text_chunks.json", 
        "vdb_entities.json",
        "vdb_relationships.json",
        "vdb_chunks.json"
    ]
    
    info["storage_files"] = {}
    for file in storage_files:
        file_path = os.path.join(WORKING_DIR, file)
        info["storage_files"][file] = {
            "exists": os.path.exists(file_path),
            "size": os.path.getsize(file_path) if os.path.exists(file_path) else 0
        }
    
    return info

def health_check() -> Dict[str, Any]:
    """Perform comprehensive health check"""
    health = {"status": "healthy", "checks": {}, "errors": []}
    
    try:
        # Check configuration
        validate_configuration()
        health["checks"]["configuration"] = "✅ Pass"
    except Exception as e:
        health["checks"]["configuration"] = f"❌ {e}"
        health["errors"].append(f"Configuration: {e}")
        health["status"] = "unhealthy"
    
    # Check embedding model
    try:
        if _hf_model is not None:
            health["checks"]["embedding_model"] = "✅ Loaded"
        else:
            health["checks"]["embedding_model"] = "⚠️ Not loaded (will load on first use)"
    except Exception as e:
        health["checks"]["embedding_model"] = f"❌ {e}"
        health["errors"].append(f"Embedding model: {e}")
    
    # Check RAG instance
    try:
        rag = get_rag()
        if rag:
            health["checks"]["rag_instance"] = "✅ Ready"
        else:
            health["checks"]["rag_instance"] = "❌ Failed to create"
            health["status"] = "unhealthy"
    except Exception as e:
        health["checks"]["rag_instance"] = f"❌ {e}"
        health["errors"].append(f"RAG instance: {e}")
        health["status"] = "unhealthy"
    
    return health