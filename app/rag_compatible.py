#!/usr/bin/env python3
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
        """Add documents to the RAG system with improved chunking."""
        if not self.embedding_model:
            print("❌ No embedding model available")
            return False
        
        print(f"📄 Processing {len(documents)} documents...")
        
        # Create better chunks with more context
        new_chunks = []
        for doc in documents:
            # Clean document text
            clean_doc = self._clean_document_text(doc)
            
            # Extract meaningful sections
            sections = self._extract_meaningful_sections(clean_doc)
            
            for section in sections:
                if len(section.strip()) > 100:  # Only substantial chunks
                    new_chunks.append(section.strip())
        
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
    
    def _clean_document_text(self, text: str) -> str:
        """Clean document text for better processing."""
        # Remove excessive whitespace and newlines
        text = ' '.join(text.split())
        
        # Fix common OCR issues
        text = text.replace('YOGYAKARTA', 'Yogyakarta')
        text = text.replace('APBD', 'APBD')
        text = text.replace('Rp.', 'Rp ')
        
        return text
    
    def _extract_meaningful_sections(self, doc: str) -> List[str]:
        """Extract meaningful sections from document with better chunking."""
        sections: List[str] = []

        import re

        # 1) Try splitting by clear headings or section markers (common in legal docs)
        #    e.g., lines in ALL CAPS, lines starting with 'KEPUTUSAN', 'BAB', or markdown '#'
        heading_pattern = re.compile(r"(?m)^(?:#\s*.+|[A-Z\s]{5,}|KEPUTUSAN|MEMUTUSKAN|PASAL)\b.*$")
        headings = heading_pattern.findall(doc)

        if headings:
            # Split by common section words first
            parts = re.split(r'(?i)(?:^|\n)\s*(KEPUTUSAN|MEMUTUSKAN|PASAL|BAB)\b', doc)
            # parts may include separators; join neighboring text into candidate sections
            candidate = ' '.join(parts)
            # fallback to paragraph split
        else:
            candidate = doc

        # 2) Split by double newlines (paragraphs) which are usually meaningful
        paragraphs = [p.strip() for p in re.split(r'\n{2,}|\r\n{2,}', candidate) if p.strip()]

        # Collect paragraphs that are substantial
        for para in paragraphs:
            if len(para) >= 120:
                sections.append(para)

        # 3) If not enough sections, split by numbered lists or bullets
        if len(sections) < 3:
            numbered = re.split(r'(?m)(?=^\s*[0-9]+[\)\.\-]\s)', candidate)
            for item in numbered:
                item = item.strip()
                if len(item) >= 120:
                    sections.append(item)

        # 4) Final fallback: sliding window chunking with overlap to ensure coverage
        if len(sections) < 5:
            # Normalize whitespace but keep sentence boundaries where possible
            text = ' '.join(candidate.split())

            max_chunk_chars = 1000
            overlap = 200
            start = 0
            text_len = len(text)

            while start < text_len:
                end = min(start + max_chunk_chars, text_len)

                # try to break at a sentence boundary within the last 200 chars
                if end < text_len:
                    window = text[start:end]
                    # find last sentence end
                    m = re.search(r'(?s)([\.\?\!])[^\.\?\!]*$', window)
                    if m and m.start() > int(max_chunk_chars * 0.3):
                        end = start + m.end()

                chunk = text[start:end].strip()
                if len(chunk) >= 100:
                    sections.append(chunk)

                if end >= text_len:
                    break

                start = max(0, end - overlap)

        # Deduplicate and ensure reasonable length
        cleaned_sections: List[str] = []
        seen = set()
        for s in sections:
            s_clean = s.strip()
            if len(s_clean) < 100:
                continue
            if s_clean in seen:
                continue
            seen.add(s_clean)
            cleaned_sections.append(s_clean)

        return cleaned_sections
    
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
    
    def _analyze_question_type(self, question: str) -> Dict[str, Any]:
        """Analyze question to provide more targeted answers."""
        question_lower = question.lower()
        
        analysis = {
            'type': 'general',
            'keywords': [],
            'expected_info': 'informasi umum'
        }
        
        # Question type detection
        if any(word in question_lower for word in ['berapa', 'jumlah', 'total', 'anggaran']):
            analysis['type'] = 'numerical'
            analysis['expected_info'] = 'angka atau nilai'
        elif any(word in question_lower for word in ['apa itu', 'definisi', 'pengertian']):
            analysis['type'] = 'definition'
            analysis['expected_info'] = 'definisi atau penjelasan'
        elif any(word in question_lower for word in ['siapa', 'who', 'pihak']):
            analysis['type'] = 'entity'
            analysis['expected_info'] = 'nama atau pihak terkait'
        elif any(word in question_lower for word in ['bagaimana', 'cara', 'proses']):
            analysis['type'] = 'process'
            analysis['expected_info'] = 'proses atau prosedur'
        
        # Extract key terms
        apbd_terms = ['apbd', 'anggaran', 'pendapatan', 'belanja', 'budget']
        location_terms = ['sleman', 'yogyakarta', 'diy']
        year_terms = ['2025', '2024']
        
        for term in apbd_terms + location_terms + year_terms:
            if term in question_lower:
                analysis['keywords'].append(term)
        
        return analysis
    
    def query(self, question: str, top_k: int = 5) -> str:
        """Query the RAG system with improved accuracy and hallucination prevention."""
        if not self.embedding_model:
            return "❌ No embedding model available for queries"
        
        if not self.chunks:
            return "❌ No documents available. Please add documents first."
        
        try:
            # Analyze question type for better targeting
            question_analysis = self._analyze_question_type(question)
            
            # Generate query embedding
            query_embedding = self.embedding_model.encode([question])[0]
            
            # Calculate similarities with better scoring
            similarities = []
            for i, chunk_embedding in enumerate(self.embeddings):
                similarity = np.dot(query_embedding, chunk_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(chunk_embedding)
                )
                
                # Boost score if chunk contains question keywords
                keyword_bonus = 0
                chunk_text = self.chunks[i].lower()
                for keyword in question_analysis['keywords']:
                    if keyword in chunk_text:
                        keyword_bonus += 0.1
                
                final_score = min(similarity + keyword_bonus, 1.0)  # Cap at 1.0
                similarities.append(final_score)
            
            # Get top-k most similar chunks with adaptive threshold
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            top_similarities = [similarities[i] for i in top_indices]
            
            # Adaptive similarity threshold based on question type
            if question_analysis['type'] == 'numerical':
                min_similarity_threshold = 0.6  # Higher for specific numbers
            elif question_analysis['type'] == 'definition':
                min_similarity_threshold = 0.5  # Medium for definitions
            else:
                min_similarity_threshold = 0.4  # Lower for general questions
            
            # Filter only high-confidence matches
            relevant_chunks = []
            relevant_scores = []
            
            for idx, score in zip(top_indices, top_similarities):
                if score >= min_similarity_threshold:
                    relevant_chunks.append(self.chunks[idx])
                    relevant_scores.append(score)
            
            if not relevant_chunks:
                return self._generate_no_match_response(question, question_analysis)
            
            # Create more accurate response
            return self._generate_structured_response(
                question, question_analysis, relevant_chunks, relevant_scores
            )
            
        except Exception as e:
            return f"❌ Error dalam query: {str(e)}"
    
    def _generate_no_match_response(self, question: str, analysis: Dict[str, Any]) -> str:
        """Generate helpful response when no good matches found."""
        response = f"❓ Maaf, saya tidak menemukan informasi yang cukup relevan untuk menjawab pertanyaan: '{question}'\\n\\n"
        
        if analysis['type'] == 'numerical':
            response += "💡 Untuk informasi angka/nilai APBD, coba tanyakan:\\n"
            response += "- 'Berapa total APBD Sleman 2025?'\\n"
            response += "- 'Berapa anggaran pendidikan Sleman 2025?'\\n"
        elif analysis['type'] == 'definition':
            response += "💡 Untuk definisi/penjelasan, coba tanyakan:\\n"
            response += "- 'Apa itu APBD?'\\n"
            response += "- 'Apa pengertian pendapatan daerah?'\\n"
        else:
            response += "💡 Coba ajukan pertanyaan yang lebih spesifik tentang:\\n"
            response += "- Anggaran dan belanja daerah\\n"
            response += "- Pendapatan daerah\\n"
            response += "- Program kerja pemerintah\\n"
            response += "- Peraturan dan kebijakan\\n"
        
        return response
    
    def _generate_structured_response(self, question: str, analysis: Dict[str, Any], 
                                    chunks: List[str], scores: List[float]) -> str:
        """Generate structured, accurate response based on question type."""
        
        # Synthesize information from chunks to create coherent response
        synthesized_info = self._synthesize_chunks(chunks, question, analysis['type'])
        
        response = f"📚 **Jawaban berdasarkan Dokumen APBD Sleman 2025:**\n\n"
        
        # Add main answer section
        if analysis['type'] == 'numerical':
            response += "🔢 **Informasi Angka/Nilai:**\n"
        elif analysis['type'] == 'definition':
            response += "📖 **Definisi/Penjelasan:**\n"
        elif analysis['type'] == 'entity':
            response += "👥 **Pihak/Instansi Terkait:**\n"
        elif analysis['type'] == 'process':
            response += "⚙️ **Proses/Prosedur:**\n"
        
        # Add synthesized answer
        response += f"{synthesized_info}\n\n"
        
        # Add supporting evidence section
        response += "📋 **Detail dan Referensi:**\n\n"
        
        # Present top 3 most relevant chunks as supporting evidence
        top_chunks = list(zip(chunks, scores))[:3]
        for i, (chunk, score) in enumerate(top_chunks, 1):
            # Clean chunk but keep more context
            clean_chunk = chunk.strip()
            
            # Extract most relevant sentences around keywords
            relevant_sentences = self._extract_relevant_sentences(clean_chunk, analysis['keywords'])
            
            # Confidence indicator
            if score >= 0.8:
                confidence = "🟢 Sangat Relevan"
            elif score >= 0.7:
                confidence = "🟡 Relevan" 
            elif score >= 0.6:
                confidence = "🟠 Cukup Relevan"
            else:
                confidence = "🔴 Mungkin Terkait"
            
            response += f"**{i}. [{confidence} - {score:.1%}]**\n"
            response += f"{relevant_sentences}\n\n"
        
        # Add metadata
        avg_confidence = sum(scores) / len(scores)
        response += "---\n"
        response += f"📊 **Informasi Tambahan:**\n"
        response += f"• Diolah dari {len(chunks)} bagian dokumen APBD Sleman 2025\n"
        response += f"• Tingkat relevansi rata-rata: {avg_confidence:.1%}\n"
        response += f"• Kata kunci: {', '.join(analysis['keywords'])}\n"
        
        if avg_confidence < 0.7:
            response += "\n⚠️ **Catatan:** Tingkat relevansi sedang. "
            response += "Silakan merujuk dokumen APBD resmi untuk informasi lengkap."
        
        return response
        
    def _synthesize_chunks(self, chunks: List[str], question: str, question_type: str) -> str:
        """Synthesize information from chunks to create a coherent answer."""
        
        # Extract key information based on question type
        if question_type == 'numerical':
            return self._extract_numerical_info(chunks, question)
        elif question_type == 'definition':
            return self._extract_definition_info(chunks, question)
        elif question_type == 'entity':
            return self._extract_entity_info(chunks, question)
        elif question_type == 'process':
            return self._extract_process_info(chunks, question)
        else:
            return self._extract_general_info(chunks, question)
    
    def _extract_numerical_info(self, chunks: List[str], question: str) -> str:
        """Extract numerical information and provide summary."""
        numbers = []
        context = []
        
        import re
        for chunk in chunks:
            # Find numbers with context - improved pattern for various formats
            number_patterns = [
                r'Rp[\d.,]+(?:\.[\d]+)?',  # Rupiah amounts
                r'[\d]+\.[\d]+\.[\d]+\.[\d]+',  # Large numbers with dots
                r'[\d]+,[\d]+(?:,[\d]+)*',  # Numbers with commas
                r'sebesar Rp[\d.,]+',  # "sebesar Rp" pattern
                r'jumlah.*Rp[\d.,]+'  # "jumlah ... Rp" pattern
            ]
            
            for pattern in number_patterns:
                matches = re.findall(pattern, chunk)
                numbers.extend(matches)
            
            # Extract sentences containing numbers with better context
            sentences = chunk.split('.')
            for sentence in sentences:
                sentence = sentence.strip()
                # Look for sentences with budget-related keywords
                if any(word in sentence.lower() for word in ['anggaran', 'apbd', 'belanja', 'pendapatan', 'total', 'jumlah', 'rp']):
                    if any(re.search(r'[\d.,]+', sentence) for _ in [True]):  # Contains numbers
                        context.append(sentence)
        
        if numbers:
            # Process and deduplicate numbers
            unique_numbers = list(set(numbers))
            
            # Sort by magnitude (rough estimation)
            def estimate_magnitude(num_str):
                # Extract digits and estimate magnitude
                digits = re.findall(r'\d+', num_str)
                if digits:
                    return len(''.join(digits))
                return 0
            
            sorted_numbers = sorted(unique_numbers, key=estimate_magnitude, reverse=True)[:3]
            
            # Check if the question asks for total/keseluruhan
            asking_total = any(word in question.lower() for word in ['total', 'keseluruhan', 'jumlah'])
            
            response = ""
            
            if asking_total:
                # Try to find the main total amount
                main_totals = []
                for ctx in context:
                    if any(word in ctx.lower() for word in ['total', 'keseluruhan', 'jumlah anggaran']):
                        main_totals.append(ctx)
                
                if main_totals:
                    response += "**Total Anggaran APBD Sleman 2025:**\n\n"
                    # Find the most relevant total context
                    best_total = max(main_totals, key=len)
                    total_numbers = re.findall(r'Rp[\d.,]+', best_total)
                    if total_numbers:
                        largest_amount = max(total_numbers, key=lambda x: len(x.replace(',', '').replace('.', '')))
                        response += f"🎯 **{largest_amount}**\n\n"
                        response += f"*Konteks:* {best_total[:200]}...\n\n"
                else:
                    response += "Berdasarkan analisis dokumen, ditemukan beberapa nilai anggaran utama:\n\n"
            else:
                response += "Berdasarkan dokumen APBD Sleman 2025:\n\n"
            
            # Add top numbers with context
            for i, num in enumerate(sorted_numbers, 1):
                # Find best context for this number
                best_context = ""
                for ctx in context:
                    if num in ctx and len(ctx) > len(best_context):
                        best_context = ctx
                
                # Determine what this number represents
                description = "Nilai anggaran"
                if best_context:
                    if 'peningkatan' in best_context.lower():
                        description = "Peningkatan anggaran"
                    elif 'belanja' in best_context.lower():
                        description = "Belanja"
                    elif 'pendapatan' in best_context.lower():
                        description = "Pendapatan"
                    elif 'total' in best_context.lower():
                        description = "Total anggaran"
                
                response += f"**{i}. {num}** - {description}\n"
                if best_context:
                    response += f"   *{best_context[:180]}...*\n\n"
                else:
                    response += "\n"
            
            return response.strip()
        
        return "Informasi numerik ditemukan dalam dokumen, namun memerlukan analisis lebih detail untuk memberikan nilai yang akurat."
    
    def _extract_definition_info(self, chunks: List[str], question: str) -> str:
        """Extract definition information."""
        relevant_sentences = []
        
        for chunk in chunks:
            sentences = chunk.split('.')
            for sentence in sentences:
                if any(keyword in sentence.lower() for keyword in ['adalah', 'merupakan', 'yaitu', 'ialah']):
                    relevant_sentences.append(sentence.strip())
        
        if relevant_sentences:
            # Take best definitions (longest and most complete)
            best_definitions = sorted(relevant_sentences, key=len, reverse=True)[:2]
            response = ""
            for i, definition in enumerate(best_definitions, 1):
                response += f"**{i}.** {definition}\n\n"
            return response.strip()
        
        return "Berdasarkan dokumen, terdapat penjelasan terkait namun memerlukan interpretasi konteks yang lebih lengkap."
    
    def _extract_entity_info(self, chunks: List[str], question: str) -> str:
        """Extract entity/organization information."""
        entities = []
        
        for chunk in chunks:
            # Look for organization patterns
            if any(word in chunk.lower() for word in ['dinas', 'badan', 'kantor', 'sekretariat', 'dprd']):
                sentences = chunk.split('.')
                for sentence in sentences:
                    if any(word in sentence.lower() for word in ['dinas', 'badan', 'kantor', 'sekretariat']):
                        entities.append(sentence.strip())
        
        if entities:
            unique_entities = list(set(entities))[:3]
            response = "Pihak/instansi yang terkait:\n\n"
            for i, entity in enumerate(unique_entities, 1):
                response += f"**{i}.** {entity[:200]}...\n\n"
            return response.strip()
        
        return "Terdapat berbagai pihak dan instansi yang terlibat dalam dokumen APBD ini."
    
    def _extract_process_info(self, chunks: List[str], question: str) -> str:
        """Extract process/procedure information."""
        process_sentences = []
        
        for chunk in chunks:
            sentences = chunk.split('.')
            for sentence in sentences:
                if any(word in sentence.lower() for word in ['tahapan', 'proses', 'prosedur', 'cara', 'langkah']):
                    process_sentences.append(sentence.strip())
        
        if process_sentences:
            response = "Proses/prosedur yang terkait:\n\n"
            for i, process in enumerate(process_sentences[:3], 1):
                response += f"**{i}.** {process}\n\n"
            return response.strip()
        
        return "Dokumen menjelaskan berbagai proses dan prosedur terkait APBD."
    
    def _extract_general_info(self, chunks: List[str], question: str) -> str:
        """Extract general information by finding most relevant sentences."""
        # Find sentences that might answer the question
        question_words = question.lower().split()
        relevant_sentences = []
        
        for chunk in chunks:
            sentences = chunk.split('.')
            for sentence in sentences:
                sentence_lower = sentence.lower()
                relevance_score = sum(1 for word in question_words if word in sentence_lower)
                if relevance_score >= 2:  # At least 2 matching words
                    relevant_sentences.append((sentence.strip(), relevance_score))
        
        if relevant_sentences:
            # Sort by relevance and take top 3
            relevant_sentences.sort(key=lambda x: x[1], reverse=True)
            top_sentences = relevant_sentences[:3]
            
            response = "Berdasarkan analisis dokumen:\n\n"
            for i, (sentence, score) in enumerate(top_sentences, 1):
                response += f"**{i}.** {sentence[:250]}...\n\n"
            return response.strip()
        
        return "Informasi terkait tersedia dalam dokumen APBD Sleman 2025."
    
    def _extract_relevant_sentences(self, chunk: str, keywords: List[str]) -> str:
        """Extract most relevant sentences from a chunk based on keywords."""
        sentences = chunk.split('.')
        relevant_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:  # Skip very short sentences
                continue
                
            # Score sentence based on keyword matches
            score = sum(1 for keyword in keywords if keyword.lower() in sentence.lower())
            if score > 0:
                relevant_sentences.append((sentence, score))
        
        if relevant_sentences:
            # Sort by relevance and take top 2
            relevant_sentences.sort(key=lambda x: x[1], reverse=True)
            top_sentences = relevant_sentences[:2]
            
            result = ""
            for sentence, score in top_sentences:
                result += f"{sentence}. "
            
            return result.strip()
        
        # If no keyword matches, return first meaningful part
        meaningful_part = chunk[:300]
        if len(chunk) > 300:
            # Find a good breaking point
            break_point = meaningful_part.rfind('.')
            if break_point > 200:
                meaningful_part = meaningful_part[:break_point + 1]
            else:
                meaningful_part += "..."
        
        return meaningful_part
    
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
        print(f"\n🔍 Testing query: {test_query}")
        response = rag.query(test_query)
        print(f"🤖 Response: {response[:200]}...")
        return True
    else:
        print("⚠️  No data available for testing")
        return False

if __name__ == "__main__":
    test_simple_rag()
