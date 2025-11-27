#!/usr/bin/env python3
"""
Debug embedding function to test async compatibility
"""
import asyncio
from app.rag_init import _create_embedding_func

async def test_embedding():
    print("Creating embedding function...")
    embedding_func = _create_embedding_func()
    
    print(f"Embedding dimension: {embedding_func.embedding_dim}")
    print(f"Max token size: {embedding_func.max_token_size}")
    
    test_texts = ["Hello world", "Testing embedding function"]
    
    print(f"Testing with texts: {test_texts}")
    
    try:
        result = await embedding_func(test_texts)
        print(f"Result type: {type(result)}")
        print(f"Result length: {len(result)}")
        print(f"First embedding shape: {len(result[0])}")
        print("✅ Embedding function works correctly!")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_embedding())
    exit(0 if success else 1)