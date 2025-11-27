import asyncio
import logging
from typing import Optional, Dict, Any, Union
from lightrag import QueryParam
from app.rag_init import get_rag
from app.config import DEFAULT_QUERY_MODE, TOP_K_ENTITIES, COSINE_THRESHOLD

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def ask(
    question: str, 
    mode: str = DEFAULT_QUERY_MODE,
    top_k: int = TOP_K_ENTITIES,
    max_retries: int = 3
) -> str:
    """
    Fungsi synchronous untuk melakukan query ke RAG system dengan fallback modes.
    
    Args:
        question: Pertanyaan yang akan diquery
        mode: Mode query (hybrid, local, global, naive)
        top_k: Jumlah entitas teratas yang akan diambil
        max_retries: Maksimal retry jika terjadi error
        
    Returns:
        str: Jawaban dari sistem RAG atau error message
    """
    try:
        # Validate input
        if not question or not question.strip():
            return "❌ Pertanyaan tidak boleh kosong."
            
        question = question.strip()
        logger.info(f"🔍 Query: {question[:100]}{'...' if len(question) > 100 else ''}")
        
        rag = get_rag()
        
        # Check if RAG instance exists and is properly initialized
        if not rag:
            return "❌ RAG system tidak dapat diinisialisasi. Silakan periksa konfigurasi."
            
        # Try different query approaches with fallback
        return _try_query_with_fallback(rag, question, mode, top_k, max_retries)
        
    except Exception as e:
        logger.error(f"❌ Unexpected error in ask(): {e}")
        return _format_error_message(str(e))

def _try_query_with_fallback(
    rag, 
    question: str, 
    preferred_mode: str, 
    top_k: int, 
    max_retries: int
) -> str:
    """Try query with multiple fallback strategies"""
    
    # Define fallback modes in order of preference
    query_modes = [preferred_mode, "local", "naive", "global"]
    # Remove duplicates while preserving order
    query_modes = list(dict.fromkeys(query_modes))
    
    last_error = None
    
    for attempt, mode in enumerate(query_modes):
        try:
            logger.info(f"🎯 Trying query mode: {mode} (attempt {attempt + 1})")
            
            # Try RAGAnything query first
            result = _query_raganything(rag, question, mode, top_k)
            if result and result.strip():
                logger.info(f"✅ Success with mode: {mode}")
                return result
                
            # Fallback to direct LightRAG if RAGAnything fails
            if hasattr(rag, 'lightrag') and rag.lightrag:
                result = _query_lightrag_direct(rag.lightrag, question, mode, top_k)
                if result and result.strip():
                    logger.info(f"✅ Success with direct LightRAG mode: {mode}")
                    return result
                    
        except Exception as e:
            last_error = e
            logger.warning(f"⚠️ Mode '{mode}' failed: {e}")
            continue
    
    # All modes failed
    logger.error(f"❌ All query modes failed. Last error: {last_error}")
    return _format_error_message(str(last_error) if last_error else "All query modes failed")

def _query_raganything(rag, question: str, mode: str, top_k: int) -> Optional[str]:
    """Query using RAGAnything interface"""
    try:
        # Use async query if available
        if hasattr(rag, 'query'):
            result = rag.query(question, mode=mode)
            return result
        return None
    except Exception as e:
        logger.debug(f"RAGAnything query failed: {e}")
        return None

def _query_lightrag_direct(lightrag, question: str, mode: str, top_k: int) -> Optional[str]:
    """Query using direct LightRAG interface"""
    try:
        param = QueryParam(mode=mode, top_k=top_k)
        if hasattr(lightrag, 'query'):
            result = lightrag.query(question, param=param)
            return result
        return None
    except Exception as e:
        logger.debug(f"Direct LightRAG query failed: {e}")
        return None

def _format_error_message(error_msg: str) -> str:
    """Format error message with helpful suggestions"""
    error_msg = error_msg.lower()
    
    if "no data found" in error_msg or "no documents" in error_msg:
        return "❌ Tidak ada data dokumen. Silakan proses PDF terlebih dahulu dengan perintah: make process"
    elif "402" in error_msg or "credits" in error_msg or "quota" in error_msg:
        return "❌ Quota API terlampaui. Silakan periksa credit API key atau gunakan model gratis."
    elif "401" in error_msg or "unauthorized" in error_msg:
        return "❌ API key tidak valid. Silakan periksa OPENROUTER_API_KEY di file .env"
    elif "async" in error_msg or "context manager" in error_msg:
        return "❌ Async error terdeteksi. Silakan restart sistem: make clean-storage && make process"
    elif "embedding" in error_msg:
        return "❌ Error embedding model. Silakan rebuild embeddings: make rebuild-embeddings"
    elif "lightrag" in error_msg or "rag" in error_msg:
        return "❌ RAG system error. Silakan re-initialize: make clean && make setup"
    else:
        return f"❌ Error: {error_msg[:200]}{'...' if len(error_msg) > 200 else ''}"

# Async wrapper functions
async def aask(
    question: str, 
    mode: str = DEFAULT_QUERY_MODE,
    top_k: int = TOP_K_ENTITIES
) -> str:
    """
    Async wrapper untuk compatibility dengan async code.
    """
    try:
        rag = get_rag()
        
        if not rag:
            return "❌ RAG system tidak dapat diinisialisasi."
            
        # Use RAGAnything async query if available
        if hasattr(rag, 'aquery'):
            result = await rag.aquery(question, mode=mode)
            if result and result.strip():
                return result
                
        # Fallback to sync version in executor
        import concurrent.futures
        loop = asyncio.get_event_loop()
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            result = await loop.run_in_executor(
                executor, 
                lambda: ask(question, mode, top_k)
            )
            return result
            
    except Exception as e:
        logger.error(f"❌ Async query error: {e}")
        return _format_error_message(str(e))

# Utility functions
def get_query_stats() -> Dict[str, Any]:
    """Get information about RAG system status"""
    try:
        rag = get_rag()
        stats = {
            "rag_initialized": rag is not None,
            "lightrag_available": hasattr(rag, 'lightrag') and rag.lightrag is not None,
            "default_mode": DEFAULT_QUERY_MODE,
            "available_modes": ["hybrid", "local", "global", "naive"]
        }
        
        if hasattr(rag, 'lightrag') and rag.lightrag:
            try:
                # Try to get storage info if available
                working_dir = getattr(rag.lightrag, 'working_dir', 'Unknown')
                stats["working_directory"] = working_dir
            except:
                pass
                
        return stats
    except Exception as e:
        return {"error": str(e), "rag_initialized": False}

def test_query_system() -> Dict[str, Any]:
    """Test query system with simple questions"""
    test_questions = [
        "Apa itu APBD?",
        "Siapa yang bertanggung jawab atas anggaran?",
        "Berapa total anggaran?"
    ]
    
    results = {"tests": [], "summary": {"passed": 0, "failed": 0}}
    
    for question in test_questions:
        try:
            result = ask(question, mode="naive", max_retries=1)
            is_success = not result.startswith("❌")
            
            test_result = {
                "question": question,
                "success": is_success,
                "response_length": len(result),
                "preview": result[:100] + "..." if len(result) > 100 else result
            }
            
            results["tests"].append(test_result)
            
            if is_success:
                results["summary"]["passed"] += 1
            else:
                results["summary"]["failed"] += 1
                
        except Exception as e:
            results["tests"].append({
                "question": question,
                "success": False,
                "error": str(e)
            })
            results["summary"]["failed"] += 1
    
    return results

# CLI and Testing Interface
def interactive_chat():
    """Interactive chat interface untuk testing"""
    print("🤖 RAG Chatbot - Interactive Mode")
    print("Ketik 'exit' untuk keluar, 'stats' untuk info system")
    print("-" * 50)
    
    while True:
        try:
            question = input("\n🔍 Pertanyaan: ").strip()
            
            if not question:
                continue
                
            if question.lower() in ['exit', 'quit', 'bye']:
                print("👋 Selamat tinggal!")
                break
                
            if question.lower() == 'stats':
                stats = get_query_stats()
                print("📊 System Stats:")
                for key, value in stats.items():
                    print(f"  {key}: {value}")
                continue
                
            if question.lower() == 'test':
                print("🧪 Running system tests...")
                test_results = test_query_system()
                print(f"✅ Tests passed: {test_results['summary']['passed']}")
                print(f"❌ Tests failed: {test_results['summary']['failed']}")
                continue
            
            print("🤖 Mencari jawaban...")
            answer = ask(question)
            print(f"💬 Jawaban: {answer}")
            
        except KeyboardInterrupt:
            print("\n👋 Chat dihentikan oleh user")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

# Main execution
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Command line mode
        question = " ".join(sys.argv[1:])
        print(f"🔍 Query: {question}")
        answer = ask(question)
        print(f"💬 Answer: {answer}")
    else:
        # Interactive mode
        try:
            # Quick system check
            print("🔧 Checking system status...")
            stats = get_query_stats()
            
            if not stats.get("rag_initialized", False):
                print("❌ RAG system not initialized. Run: make process")
                sys.exit(1)
                
            print("✅ System ready!")
            interactive_chat()
            
        except Exception as e:
            print(f"❌ System error: {e}")
            print("💡 Try: make clean && make process")
            sys.exit(1)