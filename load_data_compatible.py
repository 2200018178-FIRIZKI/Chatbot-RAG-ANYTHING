#!/usr/bin/env python3
"""
📄 Compatible Data Loader
========================

Memuat data ke Simple RAG system dari existing storage atau PDF files.

Usage:
    python load_data_compatible.py

Author: RAG-Anything-Chatbot Compatible Version
Version: 1.0.0
"""

import os
import json
from pathlib import Path
from app.rag_compatible import get_simple_rag

def load_from_existing_storage():
    """Load data from existing RAG storage."""
    print("📂 Loading from existing storage...")
    
    storage_path = Path("./rag_storage")
    
    # Try to load from various storage files
    documents = []
    
    # Load from full docs
    full_docs_file = storage_path / "kv_store_full_docs.json"
    if full_docs_file.exists():
        try:
            with open(full_docs_file, 'r', encoding='utf-8') as f:
                full_docs_data = json.load(f)
                for doc_id, doc_content in full_docs_data.items():
                    if isinstance(doc_content, str) and len(doc_content) > 100:
                        documents.append(doc_content)
                        
            print(f"📚 Loaded {len(documents)} documents from full_docs")
        except Exception as e:
            print(f"⚠️  Could not load full_docs: {e}")
    
    # Load from text chunks
    chunks_file = storage_path / "kv_store_text_chunks.json" 
    if chunks_file.exists() and not documents:
        try:
            with open(chunks_file, 'r', encoding='utf-8') as f:
                chunks_data = json.load(f)
                chunks_list = []
                for chunk_id, chunk_content in chunks_data.items():
                    if isinstance(chunk_content, str) and len(chunk_content) > 50:
                        chunks_list.append(chunk_content)
                
                # Group chunks into documents
                if chunks_list:
                    # Every 5 chunks = 1 document
                    for i in range(0, len(chunks_list), 5):
                        doc = " ".join(chunks_list[i:i+5])
                        documents.append(doc)
                        
            print(f"📄 Loaded {len(documents)} documents from chunks")
        except Exception as e:
            print(f"⚠️  Could not load chunks: {e}")
    
    return documents

def load_from_pdf_output():
    """Load data from PDF output folder."""
    print("📁 Loading from PDF output...")
    
    output_path = Path("./output")
    documents = []
    
    # Look for markdown files in output folder
    for md_file in output_path.rglob("*.md"):
        try:
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()
                if len(content) > 100:
                    documents.append(content)
            print(f"📄 Loaded: {md_file.name}")
        except Exception as e:
            print(f"⚠️  Could not load {md_file}: {e}")
    
    return documents

def main():
    """Main data loading function."""
    print("📚 Compatible Data Loader")
    print("=" * 40)
    
    # Get RAG system
    rag = get_simple_rag()
    
    # Check current state
    stats = rag.get_stats()
    print(f"Current state: {stats['total_chunks']} chunks")
    
    # Load documents
    documents = []
    
    # Try existing storage first
    docs_from_storage = load_from_existing_storage()
    if docs_from_storage:
        documents.extend(docs_from_storage)
        print(f"✅ Loaded {len(docs_from_storage)} documents from storage")
    
    # Try PDF output
    if not documents:
        docs_from_pdf = load_from_pdf_output()
        if docs_from_pdf:
            documents.extend(docs_from_pdf)
            print(f"✅ Loaded {len(docs_from_pdf)} documents from PDF output")
    
    # Load into RAG system
    if documents:
        print(f"\n🔄 Loading {len(documents)} documents into Compatible RAG...")
        success = rag.add_documents(documents)
        
        if success:
            print("✅ Documents loaded successfully!")
            
            # Show new stats
            new_stats = rag.get_stats()
            print(f"\n📊 Updated Stats:")
            for key, value in new_stats.items():
                print(f"  • {key}: {value}")
                
            # Test a query
            print("\n🧪 Testing query...")
            test_response = rag.query("Apa itu APBD?")
            print(f"Test response: {test_response[:100]}...")
            
        else:
            print("❌ Failed to load documents")
    else:
        print("❌ No documents found to load")

if __name__ == "__main__":
    main()
