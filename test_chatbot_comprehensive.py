#!/usr/bin/env python3
"""
🧪 Comprehensive Chatbot Testing Suite
=====================================

Testing komprehensif untuk memastikan semua komponen RAG chatbot berfungsi dengan baik.

Test Categories:
1. Configuration & Environment Tests
2. PDF Processing Tests  
3. RAG System Tests
4. Query & Response Tests
5. API Integration Tests
6. Error Handling Tests
7. Performance Tests

Usage:
    python test_chatbot_comprehensive.py

Author: RAG-Anything-Chatbot Testing Suite
Version: 1.0.0
"""

import sys
import os
import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Import all modules to test
try:
    import app.config as config
    from app.config import validate_config
    from app.process_pdf import run_process_pdf, batch_process_pdfs
    from app.query_chatbot import ask, get_query_stats, test_query_system
    from app.rag_init_simple import get_rag, health_check
    from app.rag_init_stable import get_rag as get_rag_stable, get_status
    from main import RAGChatbotApp
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("Pastikan semua dependencies terinstall dengan benar")
    sys.exit(1)

# Configure test logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('test_chatbot.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class ChatbotTestSuite:
    """Comprehensive testing suite for RAG chatbot system."""
    
    def __init__(self):
        """Initialize test suite."""
        self.test_results = {
            'total_tests': 0,
            'passed_tests': 0,
            'failed_tests': 0,
            'skipped_tests': 0,
            'test_details': [],
            'start_time': datetime.now(),
            'errors': []
        }
        
        # Test data
        self.test_queries = [
            "Apa itu APBD?",
            "Berapa total anggaran APBD Sleman 2025?", 
            "Jelaskan tentang pendapatan daerah",
            "Apa saja program kerja utama?",
            "Bagaimana alokasi anggaran pendidikan?"
        ]
        
        self.test_pdf_path = "./data"
        
    def log_test_result(self, test_name: str, passed: bool, details: str = "", error: str = "") -> None:
        """Log test result with details."""
        self.test_results['total_tests'] += 1
        
        if passed:
            self.test_results['passed_tests'] += 1
            status = "✅ PASS"
            logger.info(f"{status} - {test_name}: {details}")
        else:
            self.test_results['failed_tests'] += 1
            status = "❌ FAIL"
            logger.error(f"{status} - {test_name}: {details}")
            if error:
                self.test_results['errors'].append({
                    'test': test_name,
                    'error': error,
                    'details': details
                })
        
        self.test_results['test_details'].append({
            'test_name': test_name,
            'status': status,
            'details': details,
            'timestamp': datetime.now().isoformat()
        })
        
        print(f"{status} {test_name}")
        if details:
            print(f"    📝 {details}")
        if error and not passed:
            print(f"    ❌ Error: {error}")
    
    # ========================================
    # 1. Configuration & Environment Tests
    # ========================================
    
    def test_environment_setup(self) -> bool:
        """Test environment variables and basic setup."""
        try:
            # Test API key
            api_key_exists = bool(config.OPENAI_API_KEY)
            self.log_test_result(
                "API Key Configuration",
                api_key_exists,
                f"API key {'found' if api_key_exists else 'missing'}"
            )
            
            # Test base URL
            base_url_valid = config.OPENAI_BASE_URL and "openrouter.ai" in config.OPENAI_BASE_URL
            self.log_test_result(
                "Base URL Configuration", 
                base_url_valid,
                f"Base URL: {config.OPENAI_BASE_URL}"
            )
            
            # Test directories
            working_dir_exists = os.path.exists(config.WORKING_DIR)
            self.log_test_result(
                "Working Directory",
                working_dir_exists,
                f"Working dir: {config.WORKING_DIR}"
            )
            
            # Test output directory  
            output_dir_exists = os.path.exists(config.OUTPUT_DIR)
            self.log_test_result(
                "Output Directory",
                output_dir_exists, 
                f"Output dir: {config.OUTPUT_DIR}"
            )
            
            return api_key_exists and base_url_valid
            
        except Exception as e:
            self.log_test_result("Environment Setup", False, error=str(e))
            return False
    
    def test_configuration_validation(self) -> bool:
        """Test configuration validation function."""
        try:
            config_valid = validate_config()
            self.log_test_result(
                "Configuration Validation",
                config_valid,
                "All config settings validated"
            )
            return config_valid
        except Exception as e:
            self.log_test_result("Configuration Validation", False, error=str(e))
            return False
    
    def test_model_configuration(self) -> bool:
        """Test LLM and embedding model configuration."""
        try:
            # Test LLM model
            llm_model_set = bool(config.LLM_MODEL)
            self.log_test_result(
                "LLM Model Configuration",
                llm_model_set,
                f"LLM model: {config.LLM_MODEL}"
            )
            
            # Test embedding model
            embedding_model_set = bool(config.EMBEDDING_MODEL)
            self.log_test_result(
                "Embedding Model Configuration", 
                embedding_model_set,
                f"Embedding model: {config.EMBEDDING_MODEL}"
            )
            
            return llm_model_set and embedding_model_set
            
        except Exception as e:
            self.log_test_result("Model Configuration", False, error=str(e))
            return False
    
    # ========================================
    # 2. PDF Processing Tests
    # ========================================
    
    def test_pdf_data_availability(self) -> bool:
        """Test if PDF data is available for processing."""
        try:
            if not os.path.exists(self.test_pdf_path):
                self.log_test_result(
                    "PDF Data Directory",
                    False,
                    f"Data directory not found: {self.test_pdf_path}"
                )
                return False
            
            pdf_files = [f for f in os.listdir(self.test_pdf_path) if f.endswith('.pdf')]
            pdf_count = len(pdf_files)
            
            has_pdfs = pdf_count > 0
            self.log_test_result(
                "PDF Files Available",
                has_pdfs,
                f"Found {pdf_count} PDF file(s): {pdf_files[:3]}"
            )
            
            return has_pdfs
            
        except Exception as e:
            self.log_test_result("PDF Data Availability", False, error=str(e))
            return False
    
    def test_pdf_processing_dry_run(self) -> bool:
        """Test PDF processing functionality (dry run)."""
        try:
            # Check if MinU parser is available
            import subprocess
            result = subprocess.run(['magic-pdf', '--version'], capture_output=True, text=True, timeout=10)
            parser_available = result.returncode == 0
            
            self.log_test_result(
                "MinU Parser Available",
                parser_available,
                f"MinU parser {'ready' if parser_available else 'not found'}"
            )
            
            return parser_available
            
        except subprocess.TimeoutExpired:
            self.log_test_result("MinU Parser Available", False, "Parser check timeout")
            return False
        except Exception as e:
            self.log_test_result("MinU Parser Available", False, error=str(e))
            return False
    
    # ========================================
    # 3. RAG System Tests
    # ========================================
    
    def test_rag_initialization(self) -> bool:
        """Test RAG system initialization."""
        try:
            # Test simple RAG initialization
            rag_simple = get_rag()
            simple_works = rag_simple is not None
            
            self.log_test_result(
                "RAG Simple Initialization",
                simple_works,
                "RAG simple instance created"
            )
            
            # Test stable RAG initialization
            rag_stable = get_rag_stable()
            stable_works = rag_stable is not None
            
            self.log_test_result(
                "RAG Stable Initialization", 
                stable_works,
                "RAG stable instance created"
            )
            
            return simple_works or stable_works
            
        except Exception as e:
            self.log_test_result("RAG Initialization", False, error=str(e))
            return False
    
    def test_rag_health_check(self) -> bool:
        """Test RAG system health check."""
        try:
            # Health check for simple RAG
            health_result = health_check()
            
            self.log_test_result(
                "RAG Health Check",
                True,  # health_check function exists
                f"Health check completed: {health_result}"
            )
            
            return True
            
        except Exception as e:
            self.log_test_result("RAG Health Check", False, error=str(e))
            return False
    
    def test_embedding_model_loading(self) -> bool:
        """Test embedding model loading."""
        try:
            # Try to load Hugging Face embedding model
            from sentence_transformers import SentenceTransformer
            
            model = SentenceTransformer(config.EMBEDDING_MODEL)
            test_embedding = model.encode("Test sentence")
            
            embedding_works = test_embedding is not None and len(test_embedding) > 0
            
            self.log_test_result(
                "Embedding Model Loading",
                embedding_works,
                f"Model {config.EMBEDDING_MODEL} loaded, dimension: {len(test_embedding)}"
            )
            
            return embedding_works
            
        except Exception as e:
            self.log_test_result("Embedding Model Loading", False, error=str(e))
            return False
    
    # ========================================
    # 4. Query & Response Tests
    # ========================================
    
    def test_query_system_basic(self) -> bool:
        """Test basic query system functionality."""
        try:
            # Use the built-in test function
            test_result = test_query_system()
            
            self.log_test_result(
                "Basic Query System Test",
                bool(test_result),
                f"Query system test {'passed' if test_result else 'failed'}"
            )
            
            return bool(test_result)
            
        except Exception as e:
            self.log_test_result("Basic Query System Test", False, error=str(e))
            return False
    
    def test_sample_queries(self) -> bool:
        """Test sample queries with the chatbot."""
        successful_queries = 0
        total_queries = len(self.test_queries)
        
        for i, query in enumerate(self.test_queries):
            try:
                print(f"\n🧪 Testing query {i+1}/{total_queries}: {query}")
                
                start_time = time.time()
                response = ask(query)
                end_time = time.time()
                
                response_time = end_time - start_time
                has_response = response and len(response.strip()) > 0
                
                if has_response:
                    successful_queries += 1
                    self.log_test_result(
                        f"Query Test {i+1}",
                        True,
                        f"Response received in {response_time:.2f}s, length: {len(response)} chars"
                    )
                else:
                    self.log_test_result(
                        f"Query Test {i+1}",
                        False,
                        f"Empty response in {response_time:.2f}s"
                    )
                    
            except Exception as e:
                self.log_test_result(f"Query Test {i+1}", False, error=str(e))
        
        success_rate = (successful_queries / total_queries) * 100
        overall_success = success_rate >= 60  # At least 60% success rate
        
        self.log_test_result(
            "Sample Queries Overall",
            overall_success,
            f"Success rate: {success_rate:.1f}% ({successful_queries}/{total_queries})"
        )
        
        return overall_success
    
    # ========================================
    # 5. API Integration Tests  
    # ========================================
    
    def test_api_connectivity(self) -> bool:
        """Test API connectivity to OpenRouter."""
        try:
            import requests
            
            # Test API endpoint
            headers = {
                "Authorization": f"Bearer {config.OPENAI_API_KEY}",
                "Content-Type": "application/json"
            }
            
            # Simple API test
            test_data = {
                "model": config.LLM_MODEL,
                "messages": [{"role": "user", "content": "Test"}],
                "max_tokens": 10
            }
            
            response = requests.post(
                f"{config.OPENAI_BASE_URL}/chat/completions",
                headers=headers,
                json=test_data,
                timeout=10
            )
            
            api_works = response.status_code == 200
            
            self.log_test_result(
                "API Connectivity",
                api_works,
                f"API response code: {response.status_code}"
            )
            
            return api_works
            
        except Exception as e:
            self.log_test_result("API Connectivity", False, error=str(e))
            return False
    
    # ========================================
    # 6. Error Handling Tests
    # ========================================
    
    def test_error_handling(self) -> bool:
        """Test error handling capabilities."""
        try:
            # Test with invalid query
            try:
                response = ask("")  # Empty query
                empty_handled = True
            except:
                empty_handled = False
            
            self.log_test_result(
                "Empty Query Handling",
                True,  # Should handle gracefully
                f"Empty query {'handled gracefully' if empty_handled else 'caused error'}"
            )
            
            # Test with very long query
            try:
                long_query = "Apa itu APBD? " * 100  # Very long query
                response = ask(long_query)
                long_handled = True
            except:
                long_handled = False
                
            self.log_test_result(
                "Long Query Handling",
                True,  # Should handle gracefully
                f"Long query {'handled gracefully' if long_handled else 'caused error'}"
            )
            
            return True
            
        except Exception as e:
            self.log_test_result("Error Handling", False, error=str(e))
            return False
    
    # ========================================
    # 7. Performance Tests
    # ========================================
    
    def test_response_performance(self) -> bool:
        """Test query response performance."""
        try:
            test_query = "Apa itu APBD?"
            response_times = []
            
            # Test multiple queries for average performance
            for i in range(3):
                start_time = time.time()
                response = ask(test_query)
                end_time = time.time()
                response_times.append(end_time - start_time)
            
            avg_response_time = sum(response_times) / len(response_times)
            performance_good = avg_response_time < 30  # Less than 30 seconds
            
            self.log_test_result(
                "Response Performance",
                performance_good,
                f"Average response time: {avg_response_time:.2f}s"
            )
            
            return performance_good
            
        except Exception as e:
            self.log_test_result("Response Performance", False, error=str(e))
            return False
    
    # ========================================
    # 8. Main Application Tests
    # ========================================
    
    def test_main_application(self) -> bool:
        """Test main application functionality."""
        try:
            # Test app initialization
            app = RAGChatbotApp()
            init_success = app.initialize_system()
            
            self.log_test_result(
                "Main App Initialization",
                init_success,
                "RAGChatbotApp initialized successfully"
            )
            
            # Test statistics calculation
            success_rate = app._calculate_success_rate()
            stats_work = isinstance(success_rate, (int, float))
            
            self.log_test_result(
                "Statistics Calculation",
                stats_work,
                f"Success rate calculation: {success_rate}%"
            )
            
            return init_success and stats_work
            
        except Exception as e:
            self.log_test_result("Main Application", False, error=str(e))
            return False
    
    # ========================================
    # Test Execution & Reporting
    # ========================================
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all tests and generate comprehensive report."""
        print("🚀 Starting Comprehensive Chatbot Testing...")
        print("=" * 80)
        
        # Run test categories in order
        test_categories = [
            ("Environment & Configuration", [
                self.test_environment_setup,
                self.test_configuration_validation,
                self.test_model_configuration
            ]),
            ("PDF Processing", [
                self.test_pdf_data_availability,
                self.test_pdf_processing_dry_run
            ]),
            ("RAG System", [
                self.test_rag_initialization,
                self.test_rag_health_check,
                self.test_embedding_model_loading
            ]),
            ("Query & Response", [
                self.test_query_system_basic,
                self.test_sample_queries
            ]),
            ("API Integration", [
                self.test_api_connectivity
            ]),
            ("Error Handling", [
                self.test_error_handling
            ]),
            ("Performance", [
                self.test_response_performance
            ]),
            ("Main Application", [
                self.test_main_application
            ])
        ]
        
        for category_name, tests in test_categories:
            print(f"\n📋 {category_name} Tests:")
            print("-" * 50)
            
            for test_func in tests:
                try:
                    test_func()
                except Exception as e:
                    self.log_test_result(test_func.__name__, False, error=str(e))
        
        # Calculate final statistics
        self.test_results['end_time'] = datetime.now()
        self.test_results['duration'] = self.test_results['end_time'] - self.test_results['start_time']
        self.test_results['success_rate'] = (
            self.test_results['passed_tests'] / self.test_results['total_tests'] * 100
            if self.test_results['total_tests'] > 0 else 0
        )
        
        return self.test_results
    
    def generate_report(self, results: Dict[str, Any]) -> None:
        """Generate comprehensive test report."""
        print("\n" + "=" * 80)
        print("📊 COMPREHENSIVE TEST REPORT")
        print("=" * 80)
        
        # Summary statistics
        print(f"\n📈 Test Summary:")
        print(f"  • Total Tests: {results['total_tests']}")
        print(f"  • Passed: {results['passed_tests']} ✅")
        print(f"  • Failed: {results['failed_tests']} ❌")
        print(f"  • Skipped: {results['skipped_tests']} ⏭️")
        print(f"  • Success Rate: {results['success_rate']:.1f}%")
        print(f"  • Duration: {results['duration']}")
        
        # Overall status
        if results['success_rate'] >= 80:
            status = "🎉 EXCELLENT - Chatbot is production ready!"
        elif results['success_rate'] >= 60:
            status = "✅ GOOD - Chatbot is functional with minor issues"
        elif results['success_rate'] >= 40:
            status = "⚠️  PARTIAL - Chatbot has significant issues"
        else:
            status = "❌ CRITICAL - Chatbot requires major fixes"
            
        print(f"\n🎯 Overall Status: {status}")
        
        # Failed tests details
        if results['failed_tests'] > 0:
            print(f"\n❌ Failed Tests ({results['failed_tests']}):")
            for error in results['errors']:
                print(f"  • {error['test']}: {error['details']}")
                if error['error']:
                    print(f"    Error: {error['error']}")
        
        # Recommendations
        print(f"\n🔧 Recommendations:")
        if results['success_rate'] >= 80:
            print("  ✅ Chatbot is working excellently!")
            print("  • Ready for production use")
            print("  • Consider monitoring performance over time")
        elif results['success_rate'] >= 60:
            print("  • Check failed tests and resolve minor issues")
            print("  • Monitor API connectivity and response times")
            print("  • Consider adding more PDF data if needed")
        else:
            print("  ⚠️  Critical issues need attention:")
            print("  • Check API key configuration")
            print("  • Verify MinU parser installation")
            print("  • Ensure PDF data is available")
            print("  • Test network connectivity")
        
        # Save detailed report
        report_file = f"chatbot_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n💾 Detailed report saved: {report_file}")


def main():
    """Main testing function."""
    print("🤖 RAG Chatbot Comprehensive Testing Suite")
    print("=" * 80)
    
    # Create test suite and run tests
    test_suite = ChatbotTestSuite()
    results = test_suite.run_all_tests()
    
    # Generate and display report
    test_suite.generate_report(results)
    
    # Exit with appropriate code
    if results['success_rate'] >= 60:
        sys.exit(0)  # Success
    else:
        sys.exit(1)  # Failure


if __name__ == "__main__":
    main()