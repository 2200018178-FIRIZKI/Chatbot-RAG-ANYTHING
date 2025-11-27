#!/usr/bin/env python3
"""
Debug LightRAG langsung untuk mengetahui masalah async
"""
import asyncio
from app.rag_init import _create_llm_model_func, _create_embedding_func, WORKING_DIR
from lightrag import LightRAG

async def test_lightrag():
    print("Creating LightRAG instance...")
    
    try:
        lightrag = LightRAG(
            working_dir=WORKING_DIR,
            llm_model_func=_create_llm_model_func(),
            embedding_func=_create_embedding_func(),
        )
        print("✅ LightRAG created successfully!")
        
        # Test query dengan berbagai cara
        print("Testing simple query...")
        result = await lightrag.aquery("Apa itu APBD?")
        print(f"Query result: {result}")
        
        # Test query naive
        print("\nTesting naive query...")
        result2 = await lightrag.aget("Apa itu APBD?")
        print(f"Naive query result: {result2}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_lightrag())
    exit(0 if success else 1)