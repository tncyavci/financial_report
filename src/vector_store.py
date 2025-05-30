"""
Vector Database Modülü
ChromaDB kullanarak hızlı similarity search ve metadata filtering
"""

import logging
import chromadb
from chromadb.config import Settings
import numpy as np
import os
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from src.text_processor import TextChunk, EmbeddingService
import uuid

logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """Arama sonucu container'ı"""
    chunk: TextChunk
    similarity_score: float
    source_file: str
    page_number: int
    content_type: str

@dataclass
class RetrievalResult:
    """Retrieval sonucu"""
    results: List[SearchResult]
    combined_context: str
    query: str
    total_results: int

class VectorStore:
    """
    ChromaDB tabanlı vector database
    Metadata filtering ve persistence dahili destekli
    """
    
    def __init__(self, persist_directory: str = "./chroma_db"):
        """
        Args:
            persist_directory: ChromaDB persistence klasörü
        """
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self.collection_name = "financial_documents"
        
        self._initialize_client()
        logger.info(f"🗃️ ChromaDB VectorStore başlatıldı - {persist_directory}")
    
    def _initialize_client(self):
        """ChromaDB client'ını başlat"""
        try:
            # ChromaDB client'ı oluştur
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Collection'ı al veya oluştur
            try:
                self.collection = self.client.get_collection(name=self.collection_name)
                logger.info(f"✅ Mevcut collection bulundu: {self.collection_name}")
            except:
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "Turkish financial documents with tables and text"}
                )
                logger.info(f"✅ Yeni collection oluşturuldu: {self.collection_name}")
            
        except Exception as e:
            logger.error(f"❌ ChromaDB başlatılamadı: {e}")
            raise
    
    def add_documents(self, chunks: List[TextChunk]):
        """Dokümanları vector store'a ekle"""
        if not chunks:
            logger.warning("⚠️ Eklenecek chunk bulunamadı")
            return
        
        logger.info(f"📦 {len(chunks)} chunk ChromaDB'ye ekleniyor...")
        
        # Embedding'i olan chunk'ları filtrele
        valid_chunks = [chunk for chunk in chunks if chunk.embedding is not None]
        
        if not valid_chunks:
            logger.warning("⚠️ Embedding'i olan chunk bulunamadı")
            return
        
        try:
            # ChromaDB için veri hazırla
            ids = []
            embeddings = []
            documents = []
            metadatas = []
            
            for chunk in valid_chunks:
                # Unique ID oluştur
                chunk_id = f"{chunk.source_file}_{chunk.page_number}_{chunk.chunk_index}_{uuid.uuid4().hex[:8]}"
                ids.append(chunk_id)
                
                # Embedding
                embeddings.append(chunk.embedding.tolist())
                
                # Document content
                documents.append(chunk.content)
                
                # Metadata
                metadata = {
                    "source_file": chunk.source_file,
                    "page_number": chunk.page_number,
                    "chunk_index": chunk.chunk_index,
                    "content_type": chunk.content_type,
                    "language": chunk.language,
                    "char_count": chunk.char_count,
                    "chunk_id": chunk.chunk_id
                }
                
                # Ek metadata varsa ekle
                if chunk.metadata:
                    for key, value in chunk.metadata.items():
                        if isinstance(value, (str, int, float, bool)):
                            metadata[f"meta_{key}"] = value
                
                metadatas.append(metadata)
            
            # ChromaDB'ye batch olarak ekle
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
            
            logger.info(f"✅ {len(valid_chunks)} chunk ChromaDB'ye eklendi")
            
        except Exception as e:
            logger.error(f"❌ ChromaDB'ye ekleme hatası: {e}")
            raise
    
    def search(self, query_embedding: np.ndarray, n_results: int = 5, 
               source_files: Optional[List[str]] = None,
               content_types: Optional[List[str]] = None,
               page_range: Optional[Tuple[int, int]] = None) -> List[SearchResult]:
        """
        Similarity search with optional filters
        
        Args:
            query_embedding: Query embedding vektörü
            n_results: Sonuç sayısı
            source_files: Kaynak dosya filtreleri
            content_types: İçerik türü filtreleri  
            page_range: Sayfa aralığı (min, max)
        """
        try:
            # Where filtresi oluştur
            where_filter = {}
            
            if source_files:
                where_filter["source_file"] = {"$in": source_files}
            
            if content_types:
                where_filter["content_type"] = {"$in": content_types}
            
            if page_range:
                min_page, max_page = page_range
                where_filter["page_number"] = {"$gte": min_page, "$lte": max_page}
            
            # ChromaDB query
            query_params = {
                "query_embeddings": [query_embedding.tolist()],
                "n_results": n_results
            }
            
            if where_filter:
                query_params["where"] = where_filter
            
            results = self.collection.query(**query_params)
            
            # Sonuçları SearchResult'a çevir
            search_results = []
            
            if results['documents'] and results['documents'][0]:
                for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0], 
                    results['distances'][0]
                )):
                    # Similarity score hesapla (ChromaDB distance'tan)
                    similarity_score = 1 / (1 + distance)
                    
                    # TextChunk'ı yeniden oluştur
                    chunk = TextChunk(
                        content=doc,
                        chunk_id=metadata.get('chunk_id', ''),
                        source_file=metadata['source_file'],
                        page_number=metadata['page_number'],
                        chunk_index=metadata['chunk_index'],
                        content_type=metadata['content_type'],
                        language=metadata.get('language', 'tr'),
                        char_count=metadata.get('char_count', len(doc))
                    )
                    
                    result = SearchResult(
                        chunk=chunk,
                        similarity_score=similarity_score,
                        source_file=metadata['source_file'],
                        page_number=metadata['page_number'],
                        content_type=metadata['content_type']
                    )
                    search_results.append(result)
            
            logger.debug(f"🔍 ChromaDB'den {len(search_results)} sonuç bulundu")
            return search_results
            
        except Exception as e:
            logger.error(f"❌ ChromaDB search hatası: {e}")
            return []
    
    def get_collection_stats(self) -> Dict:
        """Collection istatistiklerini döndür"""
        try:
            # Collection count
            count = self.collection.count()
            
            # Metadata istatistikleri için sample al
            sample_size = min(1000, count)
            if count > 0:
                sample_results = self.collection.get(limit=sample_size)
                
                # İstatistikleri hesapla
                content_types = {}
                languages = {}
                sources = set()
                
                for metadata in sample_results['metadatas']:
                    # İçerik türü
                    content_type = metadata.get('content_type', 'unknown')
                    content_types[content_type] = content_types.get(content_type, 0) + 1
                    
                    # Dil
                    language = metadata.get('language', 'tr')
                    languages[language] = languages.get(language, 0) + 1
                    
                    # Kaynak dosya
                    sources.add(metadata.get('source_file', 'unknown'))
                
                stats = {
                    'total_documents': count,
                    'is_trained': count > 0,
                    'content_type_distribution': content_types,
                    'language_distribution': languages,
                    'unique_sources': len(sources),
                    'sample_size_for_distributions': sample_size,
                    'database_type': 'chromadb',
                    'collection_name': self.collection_name
                }
            else:
                stats = {
                    'total_documents': 0,
                    'is_trained': False,
                    'content_type_distribution': {},
                    'language_distribution': {},
                    'unique_sources': 0,
                    'sample_size_for_distributions': 0,
                    'database_type': 'chromadb',
                    'collection_name': self.collection_name
                }
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ Stats alınamadı: {e}")
            return {
                'total_documents': 0,
                'is_trained': False,
                'database_type': 'chromadb',
                'error': str(e)
            }
    
    def clear(self):
        """Collection'ı temizle"""
        try:
            # Collection'ı sil ve yeniden oluştur
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "Turkish financial documents with tables and text"}
            )
            logger.info("🗑️ ChromaDB collection temizlendi")
        except Exception as e:
            logger.error(f"❌ Collection temizlenemedi: {e}")
    
    def get_documents_by_source(self, source_file: str) -> List[Dict]:
        """Belirli bir kaynak dosyadan tüm dokümanları al"""
        try:
            results = self.collection.get(
                where={"source_file": source_file}
            )
            return results
        except Exception as e:
            logger.error(f"❌ Kaynak dosya sorgusu hatası: {e}")
            return []

class RetrievalService:
    """
    RAG için retrieval servisi
    ChromaDB ile gelişmiş query processing
    """
    
    def __init__(self, vector_store: VectorStore, embedding_service: EmbeddingService):
        """
        Args:
            vector_store: ChromaDB vector store
            embedding_service: Embedding servisi
        """
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self._query_cache = {}  # Query embedding cache for speed
        
        logger.info("🔍 ChromaDB RetrievalService başlatıldı")
    
    def retrieve_context(self, query: str, n_results: int = 5, max_context_length: int = 3000) -> RetrievalResult:
        """
        Query için context retrieve et
        
        Args:
            query: Kullanıcı sorgusu
            n_results: Kaç sonuç alınacağı
            max_context_length: Maksimum context uzunluğu
            
        Returns:
            RetrievalResult: Retrieval sonucu
        """
        logger.debug(f"🔍 Query: {query[:100]}...")
        
        try:
            # Query embedding cache kontrolü
            query_key = f"{query}_{n_results}"
            if query_key in self._query_cache:
                query_embedding = self._query_cache[query_key]
                logger.debug("⚡ Cache'den embedding alındı")
            else:
                # Query embedding'i oluştur
                query_embedding = self.embedding_service.encode([query])[0]
                self._query_cache[query_key] = query_embedding
                
                # Cache boyutu kontrolü (maksimum 50 query)
                if len(self._query_cache) > 50:
                    # En eski 10'unu sil
                    keys_to_remove = list(self._query_cache.keys())[:10]
                    for key in keys_to_remove:
                        del self._query_cache[key]
            
            # ChromaDB search
            search_results = self.vector_store.search(query_embedding, n_results)
            
            # Context'i birleştir - max_context_length kullan
            combined_context = self._combine_context(search_results, max_context_length)
            
            result = RetrievalResult(
                results=search_results,
                combined_context=combined_context,
                query=query,
                total_results=len(search_results)
            )
            
            logger.debug(f"✅ {len(search_results)} sonuç bulundu")
            return result
            
        except Exception as e:
            logger.error(f"❌ Retrieval hatası: {e}")
            return RetrievalResult(
                results=[],
                combined_context="",
                query=query,
                total_results=0
            )
    
    def search_by_filters(self, 
                         query: str, 
                         source_files: Optional[List[str]] = None,
                         content_types: Optional[List[str]] = None,
                         page_range: Optional[Tuple[int, int]] = None,
                         n_results: int = 5) -> RetrievalResult:
        """
        Filtrelere göre arama yap (ChromaDB built-in filtering)
        
        Args:
            query: Arama sorgusu
            source_files: Kaynak dosya filtreleri
            content_types: İçerik türü filtreleri
            page_range: Sayfa aralığı (min, max)
            n_results: Sonuç sayısı
        """
        try:
            # Query embedding'i oluştur
            query_embedding = self.embedding_service.encode([query])[0]
            
            # ChromaDB built-in filtering ile search
            search_results = self.vector_store.search(
                query_embedding=query_embedding,
                n_results=n_results,
                source_files=source_files,
                content_types=content_types,
                page_range=page_range
            )
            
            # Context oluştur
            combined_context = self._combine_context(search_results)
            
            return RetrievalResult(
                results=search_results,
                combined_context=combined_context,
                query=query,
                total_results=len(search_results)
            )
            
        except Exception as e:
            logger.error(f"❌ Filtered search hatası: {e}")
            return RetrievalResult(
                results=[],
                combined_context="",
                query=query,
                total_results=0
            )
    
    def _combine_context(self, search_results: List[SearchResult], max_context_length: int = 3000) -> str:
        """Search sonuçlarını birleştirip context oluştur (Optimized)"""
        if not search_results:
            return ""
        
        context_parts = []
        total_length = 0
        seen_content = set()  # Duplicate content kontrolü
        
        for i, result in enumerate(search_results):
            chunk = result.chunk
            
            # Duplicate kontrolü (ilk 50 karakter ile - daha hızlı)
            content_hash = chunk.content[:50]
            if content_hash in seen_content:
                continue
            seen_content.add(content_hash)
            
            # Simplified context format (daha hızlı)
            content = chunk.content
            if total_length + len(content) > max_context_length:
                # Kalan alanı kullan
                remaining_space = max_context_length - total_length - 50
                if remaining_space > 100:
                    content = content[:remaining_space] + "..."
                else:
                    break
            
            context_parts.append(content)
            total_length += len(content)
            
            # Early exit if max length reached
            if total_length >= max_context_length:
                break
        
        # Join once (more efficient)
        combined = "\n\n".join(context_parts)
        return combined
    
    def get_retrieval_stats(self) -> Dict:
        """Retrieval istatistiklerini döndür"""
        vector_stats = self.vector_store.get_collection_stats()
        embedding_info = self.embedding_service.get_model_info()
        
        return {
            'vector_store_stats': vector_stats,
            'embedding_model': embedding_info['model_name'],
            'embedding_dimension': embedding_info['embedding_dim'],
            'service_ready': vector_stats['is_trained'] and embedding_info['model_loaded']
        }
    
    def get_sources_summary(self) -> Dict:
        """Kaynak dosyaların özetini al"""
        try:
            stats = self.vector_store.get_collection_stats()
            
            # Her kaynak dosya için chunk sayıları
            source_chunks = {}
            if stats['total_documents'] > 0:
                # Bu kısım daha detaylı istatistik için geliştirilebilir
                pass
            
            return {
                'total_sources': stats.get('unique_sources', 0),
                'total_documents': stats['total_documents'],
                'content_distribution': stats.get('content_type_distribution', {})
            }
            
        except Exception as e:
            logger.error(f"❌ Source summary hatası: {e}")
            return {} 