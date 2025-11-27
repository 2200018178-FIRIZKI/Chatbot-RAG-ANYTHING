import asyncio
import os
import logging
from typing import Optional
from app.rag_init import get_rag, reset_rag_instance
from app.config import OUTPUT_DIR, WORKING_DIR, DEFAULT_QUERY_MODE

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def process_pdf(pdf_filename: str = "sleman_apbd_2025.pdf") -> Optional[dict]:
    """
    Fungsi untuk memproses PDF menggunakan shared RAG instance.
    
    Args:
        pdf_filename: Nama file PDF di folder data
        
    Returns:
        dict: Hasil processing atau None jika error
    """
    logger.info("🔄 Memulai parsing PDF...")
    
    try:
        # Validate paths first
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        pdf_path = os.path.join(base_path, "data", pdf_filename)
        
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"❌ PDF tidak ditemukan: {pdf_path}")
            
        logger.info(f"📄 File: {pdf_path}")
        
        # Reset instance untuk fresh start
        reset_rag_instance()
        
        # Get shared RAG instance
        rag = get_rag()
        
        logger.info("⚙️ Parser: ocr")
        logger.info("-" * 60)

        # Process PDF dengan error handling yang lebih baik
        result = await rag.process_document_complete(pdf_path)
        
        if result is None:
            logger.warning("⚠️ Processing returned None result")
            return None
            
        logger.info("✅ Parsing selesai!")
        logger.info(f"📁 Output parsing disimpan di: {OUTPUT_DIR}")
        logger.info(f"📁 Storage RAG: {WORKING_DIR}")
        
        # Test query sederhana dengan error handling
        await _test_query(rag)
        
        logger.info("✅ PDF telah diproses sepenuhnya!")
        logger.info("Sekarang kamu bisa bertanya ke chatbot 😊")
        
        return result
        
    except FileNotFoundError as e:
        logger.error(f"❌ File tidak ditemukan: {e}")
        raise
    except Exception as e:
        logger.error(f"❌ Error saat memproses PDF: {e}")
        raise

async def _test_query(rag, max_length: int = 200) -> None:
    """Helper function untuk test query dengan error handling"""
    try:
        logger.info("🔍 Contoh Query:")
        test_query = "Apa rekomendasi utama dalam dokumen APBD ini?"
        
        # Try different query modes if one fails
        for mode in [DEFAULT_QUERY_MODE, "local", "naive"]:
            try:
                test_result = await rag.aquery(test_query, mode=mode)
                if test_result:
                    result_str = str(test_result)
                    display_result = result_str[:max_length] + "..." if len(result_str) > max_length else result_str
                    logger.info(f"Query result ({mode}): {display_result}")
                    break
            except Exception as e:
                logger.warning(f"Query mode '{mode}' failed: {e}")
                continue
        else:
            logger.warning("⚠️ Semua mode query gagal, tapi processing berhasil")
            
    except Exception as e:
        logger.warning(f"⚠️ Test query gagal: {e}")
        logger.info("Processing PDF berhasil, tapi test query tidak bisa dilakukan")

def run_process_pdf(pdf_filename: str = "sleman_apbd_2025.pdf") -> Optional[dict]:
    """
    Wrapper synchronous dengan better error handling
    
    Args:
        pdf_filename: Nama file PDF di folder data
        
    Returns:
        dict: Hasil processing atau None jika error
    """
    try:
        return asyncio.run(process_pdf(pdf_filename))
    except KeyboardInterrupt:
        logger.info("❌ Proses dibatalkan oleh user")
        return None
    except FileNotFoundError as e:
        logger.error(f"❌ File Error: {e}")
        return None
    except Exception as e:
        logger.error(f"❌ Unexpected Error: {e}")
        return None

def batch_process_pdfs(pdf_filenames: list) -> dict:
    """
    Process multiple PDFs in sequence
    
    Args:
        pdf_filenames: List of PDF filenames in data folder
        
    Returns:
        dict: Results summary
    """
    results = {"success": [], "failed": []}
    
    for pdf_file in pdf_filenames:
        logger.info(f"🔄 Processing: {pdf_file}")
        result = run_process_pdf(pdf_file)
        
        if result:
            results["success"].append(pdf_file)
            logger.info(f"✅ Success: {pdf_file}")
        else:
            results["failed"].append(pdf_file)
            logger.error(f"❌ Failed: {pdf_file}")
    
    logger.info(f"📊 Summary: {len(results['success'])} success, {len(results['failed'])} failed")
    return results

if __name__ == "__main__":
    run_process_pdf()