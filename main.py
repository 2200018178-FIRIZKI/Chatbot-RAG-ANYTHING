#!/usr/bin/env python3
"""
🚀 RAGAnything Chatbot - Main Entry Point
==========================================

Enhanced main application with comprehensive error handling, system status monitoring,
interactive features, and robust configuration management.

Features:
- System health checks and diagnostics
- Interactive PDF processing with progress tracking
- Enhanced chatbot interface with conversation history
- Comprehensive error handling and recovery
- Development and debug modes
- Performance monitoring and statistics

Author: Enhanced for RAG-Anything-Chatbot
Version: 2.0.0 (Enhanced)
"""

import sys
import os
import time
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent))

import app.config as config
from app.config import validate_config
from app.query_compatible import ask_compatible

# Use compatible process function if the regular one has issues
try:
    from app.process_pdf import run_process_pdf, batch_process_pdfs
    PDF_FUNCTIONS_AVAILABLE = True
except ImportError:
    print("⚠️  Using basic PDF processing (enhanced functions not available)")
    PDF_FUNCTIONS_AVAILABLE = False
    
    def run_process_pdf():
        print("❌ PDF processing not available. Please run:")
        print("   python load_data_compatible.py")
        return False
    
    def batch_process_pdfs(data_path):
        print("❌ Batch PDF processing not available.")
        return 0

# Try to import enhanced functions, fall back to basic ones if not available
try:
    from app.query_chatbot import get_query_stats, test_query_system, interactive_chat
    ENHANCED_FUNCTIONS_AVAILABLE = True
except ImportError:
    print("⚠️  Using compatible query system (enhanced functions not available)")
    ENHANCED_FUNCTIONS_AVAILABLE = False
    
    # Create fallback functions
    def get_query_stats():
        return {"compatible_mode": True}
    
    def test_query_system():
        # Test the compatible system
        try:
            response = ask_compatible("test query")
            return response and len(response) > 0
        except:
            return False
    
    def interactive_chat():
        print("🤖 Interactive Chat Mode - Compatible System")
        print("=" * 50)
        print("Ask questions about your documents!")
        print("Type 'exit' or 'quit' to return to main menu.\n")
        
        while True:
            try:
                user_input = input("🧑 Your question: ").strip()
                
                if user_input.lower() in ['exit', 'quit', 'keluar']:
                    print("\n👋 Returning to main menu...")
                    break
                    
                if not user_input:
                    print("❓ Please ask a question.")
                    continue
                
                print("🤖 Thinking...")
                response = ask_compatible(user_input)
                print(f"\n🤖 Answer: {response}\n")
                print("-" * 60)
                
            except KeyboardInterrupt:
                print("\n\n👋 Chat interrupted. Returning to main menu...")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                print("Please try again or type 'exit' to return to main menu.\n")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rag_chatbot.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class RAGChatbotApp:
    """Enhanced main application class with comprehensive features."""
    
    def __init__(self):
        """Initialize the RAG Chatbot application."""
        self.config = self._load_config_dict()
        self.conversation_history: List[Dict[str, Any]] = []
        self.session_stats = {
            'start_time': datetime.now(),
            'queries_processed': 0,
            'successful_queries': 0,
            'failed_queries': 0,
            'pdf_files_processed': 0
        }
    
    def _load_config_dict(self) -> Dict[str, Any]:
        """Load configuration as dictionary from config module."""
        return {
            'api_key': config.OPENAI_API_KEY,
            'base_url': config.OPENAI_BASE_URL,
            'llm_model': config.LLM_MODEL,
            'embedding_model': config.EMBEDDING_MODEL,
            'embedding_dimension': config.EMBEDDING_DIMENSION,
            'max_tokens': config.MAX_TOKENS,
            'temperature': config.TEMPERATURE,
            'output_dir': config.OUTPUT_DIR,
            'parser': config.PARSER,
            'parse_method': config.PARSE_METHOD,
            'working_dir': config.WORKING_DIR,
            'rag_storage_path': config.WORKING_DIR,  # WORKING_DIR is the RAG storage path
            'data_folder': './data'  # Default data folder
        }
        
    def initialize_system(self) -> bool:
        """Initialize system configuration and perform health checks."""
        try:
            logger.info("🔧 Initializing RAG Chatbot system...")
            
            # Validate configuration
            if not validate_config():
                logger.error("❌ Configuration validation failed")
                return False
                
            logger.info("✅ Configuration loaded and validated")
            return True
            
        except Exception as e:
            logger.error(f"❌ System initialization failed: {e}")
            return False
    
    def show_system_status(self) -> None:
        """Display comprehensive system status and health information."""
        try:
            print("\n" + "=" * 60)
            print("           📊 SYSTEM STATUS & HEALTH CHECK")
            print("=" * 60)
            
            # Configuration status
            print("\n🔧 Configuration:")
            print(f"  • Config loaded: {'✅' if self.config else '❌'}")
            print(f"  • LLM Model: {self.config.get('llm_model', 'Not configured')}")
            print(f"  • Embedding Model: {self.config.get('embedding_model', 'Not configured')}")
            
            # Storage status
            storage_path = self.config.get('rag_storage_path', './rag_storage')
            storage_exists = os.path.exists(storage_path)
            print(f"  • RAG Storage: {'✅' if storage_exists else '❌'} ({storage_path})")
            
            # Data status
            data_path = self.config.get('data_folder', './data')
            data_exists = os.path.exists(data_path)
            pdf_count = len([f for f in os.listdir(data_path) if f.endswith('.pdf')]) if data_exists else 0
            print(f"  • Data Folder: {'✅' if data_exists else '❌'} ({pdf_count} PDF files)")
            
            # Session statistics
            print(f"\n📈 Session Statistics:")
            duration = datetime.now() - self.session_stats['start_time']
            print(f"  • Session Duration: {duration}")
            print(f"  • Queries Processed: {self.session_stats['queries_processed']}")
            print(f"  • Success Rate: {self._calculate_success_rate():.1f}%")
            print(f"  • PDFs Processed: {self.session_stats['pdf_files_processed']}")
            
            # System health test
            print(f"\n🧪 Quick Health Test:")
            try:
                test_result = test_query_system()
                print(f"  • Query System: {'✅' if test_result else '❌'}")
            except Exception as e:
                print(f"  • Query System: ❌ ({str(e)[:50]}...)")
                
        except Exception as e:
            logger.error(f"Error displaying system status: {e}")
            print("❌ Unable to display complete system status")
    
    def _calculate_success_rate(self) -> float:
        """Calculate query success rate percentage."""
        total = self.session_stats['queries_processed']
        if total == 0:
            return 0.0
        return (self.session_stats['successful_queries'] / total) * 100
    
    def show_main_menu(self) -> str:
        """Display the enhanced main menu and get user choice."""
        print("\n" + "=" * 60)
        print("           🚀 RAGAnything Chatbot — APBD Sleman")
        print("=" * 60)
        print("1️⃣  Proses PDF (wajib jika data belum di-load)")
        print("2️⃣  Proses Batch PDF (multiple files)")
        print("3️⃣  Mode Chatbot Interactive")
        print("4️⃣  Test Query System")
        print("5️⃣  Tampilkan Status System")
        print("6️⃣  Tampilkan Statistik Query")
        print("7️⃣  Reset Conversation History")
        print("8️⃣  Mode Debug/Development")
        print("9️⃣  Exit")
        print("=" * 60)
        
        return input("\n📝 Pilih menu (1-9): ").strip()
    
    def process_single_pdf(self) -> bool:
        """Process a single PDF with enhanced error handling and progress tracking."""
        try:
            print("\n🔄 Memproses PDF... mohon tunggu sebentar.")
            print("⏳ Proses ini mungkin memakan waktu beberapa menit...")
            
            start_time = time.time()
            run_process_pdf()
            end_time = time.time()
            
            processing_time = end_time - start_time
            self.session_stats['pdf_files_processed'] += 1
            
            print(f"\n✅ PDF telah diproses sepenuhnya!")
            print(f"⏱️  Waktu pemrosesan: {processing_time:.2f} detik")
            print("🎉 Sekarang kamu bisa bertanya ke chatbot!")
            
            logger.info(f"PDF processed successfully in {processing_time:.2f} seconds")
            return True
            
        except Exception as e:
            logger.error(f"PDF processing failed: {e}")
            print(f"\n❌ Gagal memproses PDF: {e}")
            print("🔧 Solusi yang bisa dicoba:")
            print("   • Periksa apakah file PDF ada di folder data/")
            print("   • Pastikan file PDF tidak corrupt")
            print("   • Check konfigurasi API key")
            print("   • Jalankan system status check (menu 5)")
            return False
    
    def process_batch_pdfs(self) -> bool:
        """Process multiple PDFs in batch mode."""
        try:
            data_path = self.config.get('data_folder', './data')
            if not os.path.exists(data_path):
                print("❌ Folder data tidak ditemukan")
                return False
                
            pdf_files = [f for f in os.listdir(data_path) if f.endswith('.pdf')]
            if not pdf_files:
                print("❌ Tidak ada file PDF ditemukan di folder data/")
                return False
                
            print(f"\n📁 Ditemukan {len(pdf_files)} file PDF:")
            for i, pdf in enumerate(pdf_files, 1):
                print(f"   {i}. {pdf}")
                
            confirm = input(f"\n🤔 Proses semua {len(pdf_files)} file PDF? (y/n): ").lower()
            if confirm not in ['y', 'yes']:
                print("❌ Batch processing dibatalkan")
                return False
                
            print("\n🔄 Memulai batch processing...")
            start_time = time.time()
            
            success_count = batch_process_pdfs(data_path)
            
            end_time = time.time()
            processing_time = end_time - start_time
            self.session_stats['pdf_files_processed'] += success_count
            
            print(f"\n✅ Batch processing selesai!")
            print(f"📊 Berhasil memproses: {success_count}/{len(pdf_files)} file")
            print(f"⏱️  Total waktu: {processing_time:.2f} detik")
            
            return success_count > 0
            
        except Exception as e:
            logger.error(f"Batch PDF processing failed: {e}")
            print(f"❌ Gagal memproses batch PDF: {e}")
            return False
    
    def run_chatbot_mode(self) -> None:
        """Run enhanced chatbot mode with conversation tracking."""
        print("\n" + "=" * 60)
        print("           🤖 Mode Chatbot RAGAnything")
        print("Ketik 'help' untuk bantuan, 'exit'/'quit' untuk keluar")
        print("=" * 60)
        
        # Use the enhanced interactive chat function
        try:
            interactive_chat()
        except KeyboardInterrupt:
            print("\n\n👋 Chatbot dihentikan oleh user. Sampai jumpa!")
        except Exception as e:
            logger.error(f"Chatbot mode error: {e}")
            print(f"\n❌ Error dalam mode chatbot: {e}")
    
    def show_query_statistics(self) -> None:
        """Display detailed query statistics and performance metrics."""
        try:
            print("\n" + "=" * 60)
            print("           📊 STATISTIK & PERFORMANCE METRICS")
            print("=" * 60)
            
            # Session statistics
            print("\n📈 Session Statistics:")
            duration = datetime.now() - self.session_stats['start_time']
            print(f"  • Session Duration: {duration}")
            print(f"  • Total Queries: {self.session_stats['queries_processed']}")
            print(f"  • Successful Queries: {self.session_stats['successful_queries']}")
            print(f"  • Failed Queries: {self.session_stats['failed_queries']}")
            print(f"  • Success Rate: {self._calculate_success_rate():.1f}%")
            print(f"  • PDFs Processed: {self.session_stats['pdf_files_processed']}")
            
            # Query system statistics
            try:
                query_stats = get_query_stats()
                if query_stats:
                    print(f"\n🔍 Query System Metrics:")
                    for key, value in query_stats.items():
                        print(f"  • {key}: {value}")
            except Exception as e:
                print(f"  • Query stats unavailable: {e}")
                
            # Conversation history summary
            if self.conversation_history:
                print(f"\n💬 Conversation Summary:")
                print(f"  • Total Conversations: {len(self.conversation_history)}")
                recent_conversations = self.conversation_history[-5:]
                print(f"  • Recent Queries:")
                for i, conv in enumerate(recent_conversations, 1):
                    query_preview = conv.get('query', '')[:50]
                    print(f"    {i}. {query_preview}{'...' if len(query_preview) == 50 else ''}")
            
        except Exception as e:
            logger.error(f"Error displaying statistics: {e}")
            print("❌ Unable to display complete statistics")
    
    def reset_conversation_history(self) -> None:
        """Reset conversation history and session statistics."""
        self.conversation_history.clear()
        self.session_stats.update({
            'start_time': datetime.now(),
            'queries_processed': 0,
            'successful_queries': 0,
            'failed_queries': 0
        })
        print("✅ Conversation history dan statistik session telah direset")
        logger.info("Conversation history and session stats reset")
    
    def run_debug_mode(self) -> None:
        """Run debug/development mode with enhanced diagnostics."""
        print("\n" + "=" * 60)
        print("           🐛 DEBUG/DEVELOPMENT MODE")
        print("=" * 60)
        
        print("\n🔍 Available Debug Options:")
        print("1. Test Query System")
        print("2. Check Configuration")
        print("3. Test PDF Processing")
        print("4. Check RAG Storage")
        print("5. Run System Diagnostics")
        print("6. View Logs")
        print("7. Back to Main Menu")
        
        choice = input("\n📝 Pilih debug option (1-7): ").strip()
        
        if choice == "1":
            print("\n🧪 Testing Query System...")
            try:
                result = test_query_system()
                print(f"Result: {'✅ PASS' if result else '❌ FAIL'}")
            except Exception as e:
                print(f"❌ Test failed: {e}")
                
        elif choice == "2":
            print("\n🔧 Configuration Check:")
            try:
                config_valid = validate_config(self.config)
                print(f"Configuration: {'✅ VALID' if config_valid else '❌ INVALID'}")
                print(f"Config contents: {self.config}")
            except Exception as e:
                print(f"❌ Config check failed: {e}")
                
        elif choice == "3":
            print("\n📄 Testing PDF Processing...")
            try:
                # This would be a dry run test
                data_path = self.config.get('data_folder', './data')
                pdf_files = [f for f in os.listdir(data_path) if f.endswith('.pdf')] if os.path.exists(data_path) else []
                print(f"PDF files found: {len(pdf_files)}")
                for pdf in pdf_files[:3]:  # Show first 3
                    print(f"  • {pdf}")
            except Exception as e:
                print(f"❌ PDF test failed: {e}")
                
        elif choice == "4":
            print("\n💾 Checking RAG Storage...")
            try:
                storage_path = self.config.get('rag_storage_path', './rag_storage')
                if os.path.exists(storage_path):
                    files = os.listdir(storage_path)
                    print(f"Storage files: {len(files)}")
                    for file in files[:5]:  # Show first 5
                        print(f"  • {file}")
                else:
                    print("❌ RAG storage tidak ditemukan")
            except Exception as e:
                print(f"❌ Storage check failed: {e}")
                
        elif choice == "5":
            self.show_system_status()
            
        elif choice == "6":
            print("\n📝 Recent Log Entries:")
            try:
                with open('rag_chatbot.log', 'r') as f:
                    lines = f.readlines()
                    for line in lines[-10:]:  # Show last 10 lines
                        print(f"  {line.strip()}")
            except Exception as e:
                print(f"❌ Cannot read log file: {e}")
                
        else:
            return
            
        input("\n📱 Press Enter to continue...")
    
    def run(self) -> None:
        """Main application loop with comprehensive menu system."""
        try:
            # Initialize system
            if not self.initialize_system():
                print("❌ System initialization failed. Exiting...")
                return
                
            logger.info("🚀 RAG Chatbot application started")
            print("✅ System initialized successfully!")
            
            while True:
                try:
                    choice = self.show_main_menu()
                    
                    if choice == "1":
                        self.process_single_pdf()
                        
                    elif choice == "2":
                        self.process_batch_pdfs()
                        
                    elif choice == "3":
                        self.run_chatbot_mode()
                        
                    elif choice == "4":
                        print("\n🧪 Testing Query System...")
                        try:
                            result = test_query_system()
                            print(f"Test Result: {'✅ PASS' if result else '❌ FAIL'}")
                        except Exception as e:
                            print(f"❌ Test failed: {e}")
                            
                    elif choice == "5":
                        self.show_system_status()
                        
                    elif choice == "6":
                        self.show_query_statistics()
                        
                    elif choice == "7":
                        self.reset_conversation_history()
                        
                    elif choice == "8":
                        self.run_debug_mode()
                        
                    elif choice == "9":
                        print("\n👋 Keluar dari aplikasi. Sampai jumpa!")
                        logger.info("Application shutdown by user")
                        break
                        
                    else:
                        print("❌ Pilihan tidak valid. Silakan pilih 1-9.")
                        
                except KeyboardInterrupt:
                    print("\n\n⚠️  Aplikasi dihentikan oleh user (Ctrl+C)")
                    confirm = input("Yakin ingin keluar? (y/n): ").lower()
                    if confirm in ['y', 'yes']:
                        break
                        
                except Exception as e:
                    logger.error(f"Menu processing error: {e}")
                    print(f"❌ Error: {e}")
                    print("🔄 Kembali ke menu utama...")
                    
        except Exception as e:
            logger.error(f"Critical application error: {e}")
            print(f"❌ Critical error: {e}")
        finally:
            # Cleanup and final statistics
            duration = datetime.now() - self.session_stats['start_time']
            logger.info(f"Session ended. Duration: {duration}, Queries: {self.session_stats['queries_processed']}")


def main() -> None:
    """Enhanced main entry point with comprehensive error handling."""
    try:
        app = RAGChatbotApp()
        app.run()
    except KeyboardInterrupt:
        print("\n\n👋 Aplikasi dihentikan. Sampai jumpa!")
    except Exception as e:
        logger.error(f"Application startup failed: {e}")
        print(f"❌ Failed to start application: {e}")
        print("🔧 Pastikan semua dependencies terinstall dengan benar")
        sys.exit(1)


if __name__ == "__main__":
    main()
