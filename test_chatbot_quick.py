#!/usr/bin/env python3
"""
🚀 Quick Chatbot Testing Script
==============================

Testing cepat dan mudah untuk memastikan chatbot RAG berfungsi dengan baik.

Usage:
    python test_chatbot_quick.py

Author: RAG-Anything-Chatbot Quick Test
Version: 1.0.0
"""

import sys
import os
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_basic_functionality():
    """Test basic chatbot functionality with simple checks."""
    print("🧪 Quick Chatbot Test - Basic Functionality")
    print("=" * 60)
    
    results = {
        'total': 0,
        'passed': 0,
        'failed': 0
    }
    
    def run_test(test_name, test_func):
        """Run a single test and track results."""
        results['total'] += 1
        try:
            print(f"\n🔍 Testing: {test_name}")
            success = test_func()
            if success:
                print(f"✅ PASS - {test_name}")
                results['passed'] += 1
            else:
                print(f"❌ FAIL - {test_name}")
                results['failed'] += 1
            return success
        except Exception as e:
            print(f"❌ ERROR - {test_name}: {str(e)[:100]}...")
            results['failed'] += 1
            return False
    
    # Test 1: Import semua modul
    def test_imports():
        try:
            import app.config as config
            from app.query_chatbot import ask
            from main import RAGChatbotApp
            return True
        except ImportError as e:
            print(f"Import error: {e}")
            return False
    
    # Test 2: Konfigurasi API
    def test_config():
        try:
            import app.config as config
            return bool(config.OPENAI_API_KEY)
        except:
            return False
    
    # Test 3: Inisialisasi aplikasi
    def test_app_init():
        try:
            from main import RAGChatbotApp
            app = RAGChatbotApp()
            return app.initialize_system()
        except:
            return False
    
    # Test 4: Test query sederhana
    def test_simple_query():
        try:
            from app.query_chatbot import ask
            response = ask("Apa itu APBD?")
            
            # Check if response is valid and not just error handling
            if response and len(response.strip()) > 0:
                # Check if response contains actual content, not just error messages
                error_indicators = [
                    "Query failed", "Error", "tidak ditemukan", 
                    "All query modes failed", "NoneType", "context manager"
                ]
                has_errors = any(indicator in response for indicator in error_indicators)
                
                if has_errors:
                    print(f"⚠️  Query returned error response: {response[:100]}...")
                    return False
                else:
                    print(f"✅ Valid response received: {response[:50]}...")
                    return True
            else:
                print("❌ Empty or no response received")
                return False
                
        except Exception as e:
            print(f"❌ Query test exception: {str(e)[:100]}...")
            return False
    
    # Test 5: Cek data PDF
    def test_pdf_data():
        try:
            data_path = "./data"
            if os.path.exists(data_path):
                pdf_files = [f for f in os.listdir(data_path) if f.endswith('.pdf')]
                return len(pdf_files) > 0
            return False
        except:
            return False
    
    # Run all tests
    run_test("Module Imports", test_imports)
    run_test("API Configuration", test_config)
    run_test("App Initialization", test_app_init)
    run_test("PDF Data Available", test_pdf_data)
    run_test("Simple Query", test_simple_query)
    
    # Results
    print("\n" + "=" * 60)
    print("📊 QUICK TEST RESULTS")
    print("=" * 60)
    
    success_rate = (results['passed'] / results['total']) * 100
    
    print(f"Total Tests: {results['total']}")
    print(f"Passed: {results['passed']} ✅")
    print(f"Failed: {results['failed']} ❌")
    print(f"Success Rate: {success_rate:.1f}%")
    
    if success_rate >= 80:
        print("\n🎉 EXCELLENT - Chatbot siap digunakan!")
        status = "READY"
    elif success_rate >= 60:
        print("\n✅ GOOD - Chatbot berfungsi dengan baik!")
        status = "FUNCTIONAL"
    else:
        print("\n⚠️  ISSUES - Ada masalah yang perlu diperbaiki!")
        status = "NEEDS_FIX"
        
    return status, results

def test_interactive_demo():
    """Interactive demo testing with user input."""
    print("\n🎮 Interactive Demo Test")
    print("=" * 40)
    print("Test chatbot dengan pertanyaan Anda sendiri!")
    print("Ketik 'quit' untuk keluar dari demo.\n")
    
    try:
        from app.query_chatbot import ask
        
        query_count = 0
        successful_queries = 0
        
        while True:
            user_input = input("🧑 Pertanyaan Anda: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'keluar']:
                break
                
            if not user_input:
                print("❌ Pertanyaan tidak boleh kosong!")
                continue
            
            print("🤖 Sedang berpikir...")
            start_time = time.time()
            
            try:
                response = ask(user_input)
                end_time = time.time()
                
                query_count += 1
                
                if response and len(response.strip()) > 0:
                    successful_queries += 1
                    print(f"🤖 Jawaban: {response}")
                    print(f"⏱️  Waktu respon: {end_time - start_time:.2f} detik")
                else:
                    print("❌ Tidak ada jawaban yang ditemukan.")
                    
            except Exception as e:
                print(f"❌ Error: {e}")
                
            print("-" * 50)
        
        if query_count > 0:
            success_rate = (successful_queries / query_count) * 100
            print(f"\n📊 Demo Statistics:")
            print(f"Total queries: {query_count}")
            print(f"Successful: {successful_queries}")
            print(f"Success rate: {success_rate:.1f}%")
            
    except ImportError:
        print("❌ Cannot import query_chatbot module")
    except Exception as e:
        print(f"❌ Demo error: {e}")

def main():
    """Main testing function."""
    print("🤖 RAG Chatbot Quick Testing")
    print("=" * 60)
    print("Pilih jenis testing:")
    print("1. Quick Test (Otomatis)")
    print("2. Interactive Demo") 
    print("3. Both")
    print("=" * 60)
    
    choice = input("Pilihan Anda (1/2/3): ").strip()
    
    if choice in ['1', '3']:
        status, results = test_basic_functionality()
        
        if choice == '3' and status in ['READY', 'FUNCTIONAL']:
            print("\n" + "=" * 60)
            input("📱 Tekan Enter untuk melanjutkan ke Interactive Demo...")
            test_interactive_demo()
    
    elif choice == '2':
        test_interactive_demo()
    
    else:
        print("❌ Pilihan tidak valid!")

if __name__ == "__main__":
    main()