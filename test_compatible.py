#!/usr/bin/env python3
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
            print(f"\n🔍 {test_name}")
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
    print("\n" + "=" * 40)
    print("📊 TEST RESULTS")
    print("=" * 40)
    
    success_rate = (results['passed'] / results['total_tests']) * 100
    
    print(f"Total Tests: {results['total_tests']}")
    print(f"Passed: {results['passed']} ✅")
    print(f"Failed: {results['failed']} ❌")
    print(f"Success Rate: {success_rate:.1f}%")
    
    if success_rate >= 75:
        print("\n🎉 EXCELLENT - Compatible system working!")
        status = "SUCCESS"
    elif success_rate >= 50:
        print("\n✅ GOOD - Compatible system mostly working!")
        status = "PARTIAL"
    else:
        print("\n⚠️  ISSUES - Compatible system needs fixes!")
        status = "FAILED"
    
    return status

def interactive_demo():
    """Run interactive demo with compatible system."""
    print("\n🎮 Interactive Demo - Compatible System")
    print("=" * 50)
    
    try:
        from app.query_compatible import ask_compatible
        
        print("Ask questions about your documents!")
        print("Type 'quit' to exit.\n")
        
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
            choice = input("\n🎮 Run interactive demo? (y/n): ").lower()
            if choice in ['y', 'yes']:
                interactive_demo()
                break
            elif choice in ['n', 'no']:
                break
    
    print("\n👋 Testing completed!")

if __name__ == "__main__":
    main()
