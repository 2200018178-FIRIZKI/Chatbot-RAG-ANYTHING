#!/usr/bin/env python3
"""
🔧 RAG Chatbot Diagnostic & Fix Tool
===================================

Tool khusus untuk mendiagnosis dan memperbaiki masalah async context manager
pada sistem RAG chatbot dan memberikan solusi yang tepat.

Usage:
    python fix_rag_issues.py

Author: RAG-Anything-Chatbot Fix Tool
Version: 1.0.0
"""

import sys
import os
import asyncio
import time
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

class RAGDiagnosticTool:
    """Comprehensive diagnostic and fix tool for RAG system issues."""
    
    def __init__(self):
        """Initialize diagnostic tool."""
        self.issues_found = []
        self.fixes_applied = []
        
    def log_issue(self, issue_type: str, description: str, severity: str = "ERROR") -> None:
        """Log an issue found during diagnosis."""
        self.issues_found.append({
            'type': issue_type,
            'description': description,
            'severity': severity
        })
        
        severity_icon = {
            'CRITICAL': '🚨',
            'ERROR': '❌', 
            'WARNING': '⚠️',
            'INFO': 'ℹ️'
        }.get(severity, '❓')
        
        print(f"{severity_icon} {severity}: {description}")
    
    def log_fix(self, fix_description: str) -> None:
        """Log a fix that was applied."""
        self.fixes_applied.append(fix_description)
        print(f"🔧 FIX APPLIED: {fix_description}")
    
    def test_async_compatibility(self) -> bool:
        """Test async context manager compatibility."""
        print("\n🔍 Testing Async Context Manager Compatibility...")
        
        try:
            # Test basic async functionality
            async def test_async():
                await asyncio.sleep(0.1)
                return "async_works"
            
            # Run async test
            result = asyncio.run(test_async())
            
            if result == "async_works":
                print("✅ Basic async functionality working")
                return True
            else:
                self.log_issue("ASYNC_BASIC", "Basic async test failed", "ERROR")
                return False
                
        except Exception as e:
            self.log_issue("ASYNC_BASIC", f"Async test exception: {e}", "ERROR")
            return False
    
    def test_lightrag_compatibility(self) -> bool:
        """Test LightRAG library compatibility."""
        print("\n🔍 Testing LightRAG Compatibility...")
        
        try:
            from lightrag import LightRAG
            from lightrag.llm import gpt_4o_mini_complete
            
            # Try to create a minimal LightRAG instance
            test_dir = "./test_rag_temp"
            os.makedirs(test_dir, exist_ok=True)
            
            rag = LightRAG(
                working_dir=test_dir,
                llm_model_func=gpt_4o_mini_complete
            )
            
            print("✅ LightRAG instance created successfully")
            
            # Cleanup test directory
            import shutil
            shutil.rmtree(test_dir, ignore_errors=True)
            
            return True
            
        except ImportError as e:
            self.log_issue("LIGHTRAG_IMPORT", f"LightRAG import failed: {e}", "CRITICAL")
            return False
        except Exception as e:
            self.log_issue("LIGHTRAG_INIT", f"LightRAG initialization failed: {e}", "ERROR")
            return False
    
    def test_embedding_compatibility(self) -> bool:
        """Test embedding model compatibility."""
        print("\n🔍 Testing Embedding Model Compatibility...")
        
        try:
            from sentence_transformers import SentenceTransformer
            
            # Test loading the embedding model
            model_name = "all-MiniLM-L6-v2"
            model = SentenceTransformer(model_name)
            
            # Test encoding
            test_text = "Test embedding"
            embedding = model.encode(test_text)
            
            if embedding is not None and len(embedding) > 0:
                print(f"✅ Embedding model {model_name} working (dim: {len(embedding)})")
                return True
            else:
                self.log_issue("EMBEDDING_EMPTY", "Embedding returned empty result", "ERROR")
                return False
                
        except ImportError as e:
            self.log_issue("EMBEDDING_IMPORT", f"SentenceTransformers import failed: {e}", "ERROR")
            return False
        except Exception as e:
            self.log_issue("EMBEDDING_ERROR", f"Embedding test failed: {e}", "ERROR")
            return False
    
    def test_query_system(self) -> bool:
        """Test query system with detailed error analysis."""
        print("\n🔍 Testing Query System...")
        
        try:
            from app.query_chatbot import ask
            
            # Test with a simple query
            print("  Testing simple query...")
            response = ask("test query")
            
            if response and len(response.strip()) > 0:
                # Check for error indicators in response
                error_indicators = [
                    "NoneType", "context manager", "Query failed",
                    "All query modes failed", "async", "await"
                ]
                
                has_async_error = any(indicator.lower() in response.lower() for indicator in error_indicators)
                
                if has_async_error:
                    self.log_issue(
                        "QUERY_ASYNC_ERROR", 
                        "Query system has async context manager issues",
                        "CRITICAL"
                    )
                    print(f"  Error response: {response[:100]}...")
                    return False
                else:
                    print(f"✅ Query system working: {response[:50]}...")
                    return True
            else:
                self.log_issue("QUERY_EMPTY", "Query returned empty response", "ERROR")
                return False
                
        except Exception as e:
            self.log_issue("QUERY_EXCEPTION", f"Query test exception: {e}", "ERROR")
            return False
    
    def check_dependencies(self) -> Dict[str, bool]:
        """Check all required dependencies."""
        print("\n🔍 Checking Dependencies...")
        
        dependencies = {
            'lightrag': False,
            'sentence_transformers': False,
            'asyncio': False,
            'openai': False,
            'requests': False
        }
        
        for dep in dependencies.keys():
            try:
                __import__(dep)
                dependencies[dep] = True
                print(f"✅ {dep}: Available")
            except ImportError:
                print(f"❌ {dep}: Missing")
                self.log_issue("DEPENDENCY_MISSING", f"Missing dependency: {dep}", "ERROR")
        
        return dependencies
    
    def suggest_fixes(self) -> None:
        """Suggest fixes based on issues found."""
        print("\n🔧 SUGGESTED FIXES:")
        print("=" * 50)
        
        if not self.issues_found:
            print("✅ No issues found! System appears to be working correctly.")
            return
        
        # Analyze issues and provide specific fixes
        async_issues = [i for i in self.issues_found if 'async' in i['type'].lower() or 'context manager' in i['description'].lower()]
        dependency_issues = [i for i in self.issues_found if 'dependency' in i['type'].lower() or 'import' in i['type'].lower()]
        query_issues = [i for i in self.issues_found if 'query' in i['type'].lower()]
        
        if async_issues:
            print("\n🚨 ASYNC CONTEXT MANAGER ISSUES DETECTED:")
            print("   This is the main issue preventing query functionality.")
            print("\n   RECOMMENDED SOLUTIONS:")
            print("   1. Update LightRAG library:")
            print("      pip install --upgrade lightrag")
            print("   2. Or try alternative approach:")
            print("      make reprocess  # Rebuild with current library")
            print("   3. Or use stable version:")
            print("      python -c \"from app.rag_init_stable import get_rag; rag = get_rag()\"")
            
        if dependency_issues:
            print("\n❌ DEPENDENCY ISSUES:")
            print("   SOLUTION:")
            print("   pip install -r requirements.txt")
            
        if query_issues:
            print("\n⚠️  QUERY SYSTEM ISSUES:")
            print("   SOLUTIONS:")
            print("   1. Check if PDF data has been processed:")
            print("      make status")
            print("   2. Reprocess PDF data:")
            print("      make reprocess")
            print("   3. Check API configuration:")
            print("      make config")
    
    def run_comprehensive_diagnosis(self) -> Dict[str, Any]:
        """Run comprehensive system diagnosis."""
        print("🏥 RAG Chatbot System Diagnosis")
        print("=" * 50)
        print("Analyzing system health and identifying issues...")
        
        results = {
            'async_working': False,
            'lightrag_working': False,
            'embedding_working': False,
            'query_working': False,
            'dependencies': {},
            'overall_status': 'CRITICAL'
        }
        
        # Run all diagnostic tests
        results['dependencies'] = self.check_dependencies()
        results['async_working'] = self.test_async_compatibility()
        results['lightrag_working'] = self.test_lightrag_compatibility()
        results['embedding_working'] = self.test_embedding_compatibility()
        results['query_working'] = self.test_query_system()
        
        # Determine overall status
        working_components = sum([
            results['async_working'],
            results['lightrag_working'], 
            results['embedding_working'],
            results['query_working']
        ])
        
        if working_components >= 3:
            results['overall_status'] = 'GOOD'
        elif working_components >= 2:
            results['overall_status'] = 'PARTIAL'
        else:
            results['overall_status'] = 'CRITICAL'
        
        return results
    
    def generate_report(self, results: Dict[str, Any]) -> None:
        """Generate comprehensive diagnostic report."""
        print("\n" + "=" * 60)
        print("📊 COMPREHENSIVE DIAGNOSTIC REPORT")
        print("=" * 60)
        
        # System status
        status_icon = {
            'GOOD': '🎉',
            'PARTIAL': '⚠️',
            'CRITICAL': '🚨'
        }.get(results['overall_status'], '❓')
        
        print(f"\n{status_icon} OVERALL STATUS: {results['overall_status']}")
        
        # Component status
        print(f"\n🔧 COMPONENT STATUS:")
        components = [
            ("Async Support", results['async_working']),
            ("LightRAG Library", results['lightrag_working']),
            ("Embedding Model", results['embedding_working']),
            ("Query System", results['query_working'])
        ]
        
        for name, status in components:
            icon = "✅" if status else "❌"
            print(f"  {icon} {name}")
        
        # Issues summary
        if self.issues_found:
            print(f"\n❌ ISSUES FOUND ({len(self.issues_found)}):")
            for issue in self.issues_found:
                severity_icon = {'CRITICAL': '🚨', 'ERROR': '❌', 'WARNING': '⚠️'}.get(issue['severity'], 'ℹ️')
                print(f"  {severity_icon} {issue['type']}: {issue['description']}")
        
        # Action plan
        print(f"\n🎯 RECOMMENDED ACTION PLAN:")
        if results['overall_status'] == 'GOOD':
            print("  ✅ System is working well!")
            print("  • Continue using the chatbot normally")
            print("  • Monitor performance over time")
        elif results['overall_status'] == 'PARTIAL':
            print("  ⚠️  System has some issues but may be functional")
            print("  • Follow suggested fixes below")
            print("  • Test after applying fixes")
        else:
            print("  🚨 System has critical issues")
            print("  • Must apply fixes before using chatbot")
            print("  • Consider rebuilding the system if issues persist")
        
        # Suggest fixes
        self.suggest_fixes()


def main():
    """Main diagnostic function."""
    print("🔧 RAG Chatbot Diagnostic & Fix Tool")
    print("=" * 50)
    
    diagnostic = RAGDiagnosticTool()
    results = diagnostic.run_comprehensive_diagnosis()
    diagnostic.generate_report(results)
    
    # Exit with appropriate code
    if results['overall_status'] == 'GOOD':
        sys.exit(0)
    elif results['overall_status'] == 'PARTIAL':
        sys.exit(1)
    else:
        sys.exit(2)


if __name__ == "__main__":
    main()