import os

from raganything import RAGAnything, RAGAnythingConfig
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc

from app.config import (
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    WORKING_DIR,
)

_rag_instance = None


# =====================================================
# 1. LLM TEXT MODEL (GPT-4o-mini via OpenRouter)
# =====================================================
def _create_llm_model_func():
    def llm_model_func(prompt, system_prompt=None, history_messages=None, **kwargs):
        if history_messages is None:
            history_messages = []

        return openai_complete_if_cache(
            "openai/gpt-4o-mini",
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL,
            max_tokens=200,
            temperature=0.3,
        )

    return llm_model_func


# =====================================================
# 2. EMBEDDING MODEL
# =====================================================
def _create_embedding_func():
    return EmbeddingFunc(
        embedding_dim=3072,
        max_token_size=8192,
        func=lambda texts: openai_embed(
            texts,
            model="text-embedding-3-large",
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL,
        ),
    )


# =====================================================
# 3. RAGAnything — SINGLETON
# =====================================================
def get_rag():
    global _rag_instance
    if _rag_instance is not None:
        return _rag_instance

    os.makedirs(WORKING_DIR, exist_ok=True)

    # Multimodal OFF
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

        # (⭐ WAJIB) override model extraction supaya TIDAK pakai GPT-4o
        extraction_model="openai/gpt-4o-mini",
        relation_model="openai/gpt-4o-mini",
        entity_model="openai/gpt-4o-mini",
        query_model="openai/gpt-4o-mini",

        # (⭐ WAJIB) batasi token internal LightRAG
        extraction_llm_max_new_tokens=200,
        relation_llm_max_new_tokens=200,
        entity_llm_max_new_tokens=200,
        query_llm_max_new_tokens=200,
    )

    # =====================================================================
    # PATCH tambahan untuk mastiin limit (anti override internal LightRAG)
    # =====================================================================
    lr = _rag_instance.lightrag

    lr.query_llm_max_new_tokens = 200
    lr.extraction_llm_max_new_tokens = 200
    lr.relation_llm_max_new_tokens = 200
    lr.entity_llm_max_new_tokens = 200

    original_run_llm = lr._run_llm

    async def patched_run_llm(*args, **kwargs):
        kwargs["max_tokens"] = 200
        kwargs["temperature"] = 0.3
        return await original_run_llm(*args, **kwargs)

    lr._run_llm = patched_run_llm
    # =====================================================================

    return _rag_instance
