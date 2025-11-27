#!/usr/bin/env python3
"""
🤖 Compatible Query System
=========================

Query system yang menggunakan Simple RAG implementation tanpa dependency issues.

Author: RAG-Anything-Chatbot Compatible Version
Version: 1.0.0
"""

import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.rag_compatible import get_simple_rag

logger = logging.getLogger(__name__)

def ask_compatible(question: str) -> str:
    """Ask question using compatible RAG system."""
    try:
        rag = get_simple_rag()
        response = rag.query(question)
        return response
    except Exception as e:
        logger.error(f"Compatible query failed: {e}")
        return f"❌ Error: {str(e)}"

def test_compatible_system():
    """Test the compatible query system."""
    print("🧪 Testing Compatible Query System...")
    
    # Test questions
    test_questions = [
        "Apa itu APBD?",
        "Berapa anggaran pendidikan?",
        "Jelaskan tentang pendapatan daerah"
    ]
    
    for question in test_questions:
        print(f"\n🔍 Q: {question}")
        response = ask_compatible(question)
        print(f"🤖 A: {response[:100]}...")
        
    return True

if __name__ == "__main__":
    test_compatible_system()
