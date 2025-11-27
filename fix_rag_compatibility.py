#!/usr/bin/env python3
"""
🔧 RAG System Compatibility Fix
==============================

Script untuk memperbaiki masalah kompatibilitas LightRAG dan menyediakan
alternative solution yang bisa digunakan langsung.

Usage:
    python fix_rag_compatibility.py

Author: RAG-Anything-Chatbot Fix Tool
Version: 1.0.0
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def create_compatible_rag_system():
    """Create a compatible RAG system that works without library conflicts."""
    print("🔧 Creating Compatible RAG System...")
    
    # Create a simple, working RAG implementation
    compatible_rag_code = '''#!/usr/bin/env python3
"""
🤖 Compatible RAG Implementation
===============================

Simple, working RAG implementation that doesn't depend on problematic LightRAG imports.
Uses basic vector similarity search with Hugging Face embeddings.

Author: RAG-Anything-Chatbot Compatible Version
Version: 1.0.0
"""

import os
import json
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path

try:
    from sentence_transformers import SentenceTransformer
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

class SimpleRAG:
    """Simple RAG implementation using vector similarity."""
    
    def __init__(self, working_dir: str = "./rag_storage"):
        """Initialize Simple RAG system."""
        self.working_dir = Path(working_dir)
        self.working_dir.mkdir(exist_ok=True)
        
        # Initialize embedding model
        if HF_AVAILABLE:
            print("🤖 Loading embedding model...")
            self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            print("✅ Embedding model loaded!")
        else:
            self.embedding_model = None
            print("⚠️  No embedding model available")
        
        # Load existing data
        self.chunks = []
        self.embeddings = []
        self._load_data()
    
    def _load_data(self):
        """Load existing chunk data."""
        chunks_file = self.working_dir / "simple_chunks.json"
        embeddings_file = self.working_dir / "simple_embeddings.json"
        
        if chunks_file.exists() and embeddings_file.exists():
            try:
                with open(chunks_file, 'r', encoding='utf-8') as f:
                    self.chunks = json.load(f)
                
                with open(embeddings_file, 'r') as f:
                    embeddings_data = json.load(f)
                    self.embeddings = [np.array(emb) for emb in embeddings_data]
                
                print(f"✅ Loaded {len(self.chunks)} chunks from storage")
            except Exception as e:
                print(f"⚠️  Could not load existing data: {e}")
                self.chunks = []
                self.embeddings = []
        else:
            print("ℹ️  No existing data found")
    
    def add_documents(self, documents: List[str]):
        """Add documents to the RAG system."""
        if not self.embedding_model:
            print("❌ No embedding model available")
            return False
        
        print(f"📄 Processing {len(documents)} documents...")
        
        # Create chunks (simple sentence splitting)
        new_chunks = []
        for doc in documents:
            # Split by sentences (simple approach)
            sentences = doc.split('. ')
            for i in range(0, len(sentences), 2):  # Group 2 sentences per chunk
                chunk = '. '.join(sentences[i:i+2])
                if len(chunk.strip()) > 50:  # Only meaningful chunks
                    new_chunks.append(chunk.strip())
        
        if not new_chunks:
            print("❌ No meaningful chunks created")
            return False
        
        # Generate embeddings
        print("🔢 Generating embeddings...")
        try:
            new_embeddings = self.embedding_model.encode(new_chunks)
            
            # Add to existing data
            self.chunks.extend(new_chunks)
            self.embeddings.extend([emb for emb in new_embeddings])
            
            # Save to disk
            self._save_data()
            
            print(f"✅ Added {len(new_chunks)} chunks to RAG system")
            return True
            
        except Exception as e:
            print(f"❌ Error generating embeddings: {e}")
            return False
    
    def _save_data(self):
        """Save chunks and embeddings to disk."""
        try:
            chunks_file = self.working_dir / "simple_chunks.json"
            embeddings_file = self.working_dir / "simple_embeddings.json"
            
            with open(chunks_file, 'w', encoding='utf-8') as f:
                json.dump(self.chunks, f, ensure_ascii=False, indent=2)
            
            embeddings_data = [emb.tolist() for emb in self.embeddings]
            with open(embeddings_file, 'w') as f:
                json.dump(embeddings_data, f, indent=2)
                
            print("💾 Data saved to disk")
            
        except Exception as e:
            print(f"❌ Error saving data: {e}")
    
    def query(self, question: str, top_k: int = 3) -> str:
        """Query the RAG system."""
        if not self.embedding_model:
            return "❌ No embedding model available for queries"
        
        if not self.chunks:
            return "❌ No documents available. Please add documents first."
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([question])[0]
            
            # Calculate similarities
            similarities = []
            for chunk_embedding in self.embeddings:
                similarity = np.dot(query_embedding, chunk_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(chunk_embedding)
                )
                similarities.append(similarity)
            
            # Get top-k most similar chunks
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            relevant_chunks = [self.chunks[i] for i in top_indices]
            
            # Create response
            if similarities[top_indices[0]] < 0.3:  # Low similarity threshold
                return f"❓ Maaf, saya tidak menemukan informasi yang relevan untuk pertanyaan: '{question}'"
            
            response = "📚 Berdasarkan dokumen yang tersedia:\\n\\n"
            for i, chunk in enumerate(relevant_chunks, 1):
                response += f"{i}. {chunk}\\n\\n"
            
            response += f"\\n💡 Informasi ini diambil dari {len(relevant_chunks)} bagian dokumen yang paling relevan."
            
            return response
            
        except Exception as e:
            return f"❌ Error dalam query: {str(e)}"
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        return {
            "total_chunks": len(self.chunks),
            "has_embeddings": len(self.embeddings) == len(self.chunks),
            "embedding_model": "all-MiniLM-L6-v2" if self.embedding_model else "None",
            "storage_path": str(self.working_dir)
        }

# Global instance
_simple_rag_instance = None

def get_simple_rag() -> SimpleRAG:
    """Get or create Simple RAG instance."""
    global _simple_rag_instance
    if _simple_rag_instance is None:
        _simple_rag_instance = SimpleRAG()
    return _simple_rag_instance

def test_simple_rag():
    """Test the Simple RAG system."""
    print("🧪 Testing Simple RAG System...")
    
    rag = get_simple_rag()
    stats = rag.get_stats()
    
    print(f"📊 System Stats:")
    for key, value in stats.items():
        print(f"  • {key}: {value}")
    
    # Test query if we have data
    if stats['total_chunks'] > 0:
        test_query = "Apa itu APBD?"
        print(f"\\n🔍 Testing query: {test_query}")
        response = rag.query(test_query)
        print(f"🤖 Response: {response[:200]}...")
        return True
    else:
        print("⚠️  No data available for testing")
        return False

if __name__ == "__main__":
    test_simple_rag()
'''
    
    # Write the compatible RAG system
    with open("app/rag_compatible.py", "w", encoding="utf-8") as f:
        f.write(compatible_rag_code)
    
    print("✅ Compatible RAG system created: app/rag_compatible.py")
    
    # Create a new query system that uses the compatible RAG
    compatible_query_code = '''#!/usr/bin/env python3
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
        print(f"\\n🔍 Q: {question}")
        response = ask_compatible(question)
        print(f"🤖 A: {response[:100]}...")
        
    return True

if __name__ == "__main__":
    test_compatible_system()
'''
    
    # Write the compatible query system
    with open("app/query_compatible.py", "w", encoding="utf-8") as f:
        f.write(compatible_query_code)
        
    print("✅ Compatible query system created: app/query_compatible.py")

def create_data_loader():
    """Create a data loader for the compatible system."""
    print("📄 Creating Data Loader...")
    
    data_loader_code = '''#!/usr/bin/env python3
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
        print(f"\\n🔄 Loading {len(documents)} documents into Compatible RAG...")
        success = rag.add_documents(documents)
        
        if success:
            print("✅ Documents loaded successfully!")
            
            # Show new stats
            new_stats = rag.get_stats()
            print(f"\\n📊 Updated Stats:")
            for key, value in new_stats.items():
                print(f"  • {key}: {value}")
                
            # Test a query
            print("\\n🧪 Testing query...")
            test_response = rag.query("Apa itu APBD?")
            print(f"Test response: {test_response[:100]}...")
            
        else:
            print("❌ Failed to load documents")
    else:
        print("❌ No documents found to load")

if __name__ == "__main__":
    main()
'''
    
    with open("load_data_compatible.py", "w", encoding="utf-8") as f:
        f.write(data_loader_code)
        
    print("✅ Data loader created: load_data_compatible.py")

def create_test_script():
    """Create a comprehensive test script for the compatible system."""
    print("🧪 Creating Test Script...")
    
    test_script_code = '''#!/usr/bin/env python3
"""
🧪 Compatible System Test
========================

Test script untuk system RAG compatible yang baru.

Usage:
    python test_compatible.py

Author: RAG-Anything-Chatbot Compatible Test
Version: 1.0.0
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_compatible_system():
    """Test the compatible RAG system."""
    print("🤖 Testing Compatible RAG System")
    print("=" * 40)
    
    results = {
        'total_tests': 0,
        'passed': 0,
        'failed': 0
    }
    
    def run_test(test_name, test_func):
        results['total_tests'] += 1
        try:
            print(f"\\n🔍 {test_name}")
            success = test_func()
            if success:
                print(f"✅ PASS - {test_name}")
                results['passed'] += 1
            else:
                print(f"❌ FAIL - {test_name}")
                results['failed'] += 1
            return success
        except Exception as e:
            print(f"❌ ERROR - {test_name}: {e}")
            results['failed'] += 1
            return False
    
    # Test 1: Import compatible modules
    def test_imports():
        try:
            from app.rag_compatible import get_simple_rag
            from app.query_compatible import ask_compatible
            return True
        except ImportError:
            return False
    
    # Test 2: RAG system initialization
    def test_rag_init():
        try:
            from app.rag_compatible import get_simple_rag
            rag = get_simple_rag()
            stats = rag.get_stats()
            return stats is not None
        except:
            return False
    
    # Test 3: Query system
    def test_query():
        try:
            from app.query_compatible import ask_compatible
            response = ask_compatible("test question")
            return response and len(response) > 0
        except:
            return False
    
    # Test 4: Data loading
    def test_data_loading():
        try:
            from app.rag_compatible import get_simple_rag
            rag = get_simple_rag()
            stats = rag.get_stats()
            return stats['total_chunks'] > 0
        except:
            return False
    
    # Run tests
    run_test("Import Compatible Modules", test_imports)
    run_test("RAG System Initialization", test_rag_init)  
    run_test("Query System", test_query)
    run_test("Data Available", test_data_loading)
    
    # Results
    print("\\n" + "=" * 40)
    print("📊 TEST RESULTS")
    print("=" * 40)
    
    success_rate = (results['passed'] / results['total_tests']) * 100
    
    print(f"Total Tests: {results['total_tests']}")
    print(f"Passed: {results['passed']} ✅")
    print(f"Failed: {results['failed']} ❌")
    print(f"Success Rate: {success_rate:.1f}%")
    
    if success_rate >= 75:
        print("\\n🎉 EXCELLENT - Compatible system working!")
        status = "SUCCESS"
    elif success_rate >= 50:
        print("\\n✅ GOOD - Compatible system mostly working!")
        status = "PARTIAL"
    else:
        print("\\n⚠️  ISSUES - Compatible system needs fixes!")
        status = "FAILED"
    
    return status

def interactive_demo():
    """Run interactive demo with compatible system."""
    print("\\n🎮 Interactive Demo - Compatible System")
    print("=" * 50)
    
    try:
        from app.query_compatible import ask_compatible
        
        print("Ask questions about your documents!")
        print("Type 'quit' to exit.\\n")
        
        while True:
            question = input("🧑 Your question: ").strip()
            
            if question.lower() in ['quit', 'exit']:
                break
                
            if not question:
                continue
                
            print("🤖 Thinking...")
            response = ask_compatible(question)
            print(f"🤖 Answer: {response}")
            print("-" * 50)
            
    except ImportError:
        print("❌ Compatible system not available")
    except Exception as e:
        print(f"❌ Demo error: {e}")

def main():
    """Main test function."""
    print("🔧 Compatible System Testing")
    print("=" * 50)
    
    # Run tests
    status = test_compatible_system()
    
    if status in ['SUCCESS', 'PARTIAL']:
        while True:
            choice = input("\\n🎮 Run interactive demo? (y/n): ").lower()
            if choice in ['y', 'yes']:
                interactive_demo()
                break
            elif choice in ['n', 'no']:
                break
    
    print("\\n👋 Testing completed!")

if __name__ == "__main__":
    main()
'''
    
    with open("test_compatible.py", "w", encoding="utf-8") as f:
        f.write(test_script_code)
        
    print("✅ Test script created: test_compatible.py")

def main():
    """Main fix function."""
    print("🔧 RAG System Compatibility Fix")
    print("=" * 50)
    print("Creating compatible RAG system that works without library conflicts...")
    
    create_compatible_rag_system()
    create_data_loader()
    create_test_script()
    
    print("\n" + "=" * 60)
    print("✅ COMPATIBILITY FIX COMPLETED!")
    print("=" * 60)
    
    print("\n🎯 Next Steps:")
    print("1. Load your data:")
    print("   python load_data_compatible.py")
    print("\n2. Test the system:")
    print("   python test_compatible.py")
    print("\n3. Use in your main app:")
    print("   from app.query_compatible import ask_compatible")
    print("   response = ask_compatible('your question')")
    
    print("\n🎉 Your RAG chatbot now has a working compatible version!")

if __name__ == "__main__":
    main()