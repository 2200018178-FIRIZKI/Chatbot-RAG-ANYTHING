import os
from raganything import RAGAnything, RAGAnythingConfig
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc

from app.config import (
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    WORKING_DIR,
)

_rag_instance = None

# =====================================================
# 1. LLM TEXT MODEL (Meta Llama 3.1 8B via OpenRouter)
# =====================================================
def _create_llm_model_func():
    def llm_model_func(prompt, system_prompt=None, history_messages=None, **kwargs):
        if history_messages is None:
            history_messages = []

        # Paksa override max_tokens untuk semua panggilan
        kwargs['max_tokens'] = 150
        kwargs['temperature'] = 0.3
        
        return openai_complete_if_cache(
            "meta-llama/llama-3.1-8b-instruct",
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL,
            **kwargs
        )

    return llm_model_func

# =====================================================
# 2. EMBEDDING MODEL - FREE HUGGING FACE
# =====================================================
def _create_embedding_func():
    """Create async embedding function using Hugging Face sentence-transformers (FREE)"""
    
    # Initialize model once at module level to avoid reloading
    global _hf_model
    if '_hf_model' not in globals():
        try:
            from sentence_transformers import SentenceTransformer
            print("🤖 Loading Hugging Face embedding model (all-MiniLM-L6-v2)...")
            _hf_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("✅ Embedding model loaded successfully!")
        except Exception as e:
            print(f"❌ Failed to load embedding model: {e}")
            _hf_model = None
    
    async def async_huggingface_embedding_func(texts):
        """Async wrapper for free local embedding using sentence-transformers"""
        if isinstance(texts, str):
            texts = [texts]
        
        try:
            if _hf_model is None:
                import numpy as np
                return [np.zeros(384).tolist() for _ in texts]
            
            # Use thread pool executor for synchronous embedding function
            import asyncio
            import concurrent.futures
            
            def encode_texts(text_list):
                return _hf_model.encode(text_list)
            
            # Run in thread pool to avoid blocking the event loop
            with concurrent.futures.ThreadPoolExecutor() as executor:
                embeddings = await asyncio.get_event_loop().run_in_executor(
                    executor, encode_texts, texts
                )
            
            # Convert to list format expected by LightRAG
            return [emb.tolist() for emb in embeddings]
            
        except Exception as e:
            print(f"Hugging Face embedding error: {e}")
            # Return zero vectors as fallback
            import numpy as np
            return [np.zeros(384).tolist() for _ in texts]
    
    return EmbeddingFunc(
        embedding_dim=384,  # all-MiniLM-L6-v2 dimension
        max_token_size=512,  # Model max token limit
        func=async_huggingface_embedding_func,
    )

# =====================================================
# 3. RAGAnything — SINGLETON SIMPLE
# =====================================================
def get_rag():
    global _rag_instance
    if _rag_instance is not None:
        return _rag_instance

    os.makedirs(WORKING_DIR, exist_ok=True)

    # Create LightRAG instance if storage exists
    lightrag_instance = None
    if os.path.exists(os.path.join(WORKING_DIR, "kv_store_full_docs.json")):
        try:
            lightrag_instance = LightRAG(
                working_dir=WORKING_DIR,
                llm_model_func=_create_llm_model_func(),
                embedding_func=_create_embedding_func(),
                embedding_func_max_async=1,  # Limit async workers
                llm_model_max_async=1,       # Limit LLM async workers
            )
            print("INFO: LightRAG instance created from existing storage")
        except Exception as e:
            print(f"Warning: Could not initialize LightRAG from storage: {e}")

    config = RAGAnythingConfig(
        working_dir=WORKING_DIR,
        parser="mineru",
        parse_method="ocr",
        enable_image_processing=False,
        enable_table_processing=False,
        enable_equation_processing=False,
    )

    llm_model_func = _create_llm_model_func()
    embedding_func = _create_embedding_func()

    _rag_instance = RAGAnything(
        config=config,
        llm_model_func=llm_model_func,
        embedding_func=embedding_func,
        lightrag=lightrag_instance  # Pass pre-initialized LightRAG
    )

    return _rag_instance

def reset_rag_instance():
    global _rag_instance
    _rag_instance = None