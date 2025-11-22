import asyncio
import os

from raganything import RAGAnything, RAGAnythingConfig
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc

from app.config import (
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    PARSE_METHOD,
    WORKING_DIR,
    OUTPUT_DIR,
)


async def process_pdf():

    # -----------------------------
    # RAGAnything TANPA MULTIMODAL
    # -----------------------------
    config = RAGAnythingConfig(
        working_dir=WORKING_DIR,
        parser="mineru",
        parse_method=PARSE_METHOD,
        enable_image_processing=False,
        enable_table_processing=False,
        enable_equation_processing=False,
    )

    # -----------------------------
    # LLM MODEL (TEXT ONLY) — Aman Token
    # -----------------------------
    def llm_model_func(prompt, system_prompt=None, history_messages=None, **kwargs):
        if history_messages is None:
            history_messages = []

        # Patch global dari rag_init.py akan memaksa max_tokens kecil
        return openai_complete_if_cache(
            "openai/gpt-4o-mini",
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL,
            **kwargs
        )

    # -----------------------------
    # Embedding Model
    # -----------------------------
    embedding_func = EmbeddingFunc(
        embedding_dim=3072,
        max_token_size=8192,
        func=lambda texts: openai_embed(
            texts,
            model="text-embedding-3-large",
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL,
        ),
    )

    # -----------------------------
    # INIT RAGAnything (NO VISION!)
    # -----------------------------
    rag = RAGAnything(
        config=config,
        llm_model_func=llm_model_func,
        embedding_func=embedding_func,
    )

    # Path PDF
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pdf_path = os.path.join(base_path, "data", "sleman_apbd_2025.pdf")

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"❌ PDF tidak ditemukan: {pdf_path}")

    print("🔄 Memulai parsing PDF...")
    print(f"📄 File: {pdf_path}")
    print(f"⚙️ Parser: {PARSE_METHOD}")
    print("-" * 60)

    # -----------------------------
    # PROCESS PDF (TEXT ONLY)
    # -----------------------------
    await rag.process_document_complete(
        file_path=pdf_path,
        output_dir=OUTPUT_DIR,
        parse_method=PARSE_METHOD,
        display_stats=True,
    )

    print("\n✅ Parsing selesai!")
    print(f"📁 Output parsing disimpan di: {OUTPUT_DIR}")
    print(f"📁 Storage RAG: {WORKING_DIR}")

    # -----------------------------
    # Contoh Query (RINGAN)
    # -----------------------------
    print("\n🔍 Contoh Query:")
    result = await rag.aquery(
        "Apa rekomendasi utama dalam dokumen APBD ini?",
        mode="local",          # 🔥 WAJIB DIGANTI — hemat token, aman OpenRouter
    )
    print(result)


def run_process_pdf():
    asyncio.run(process_pdf())


if __name__ == "__main__":
    run_process_pdf()
