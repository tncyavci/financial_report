"""
Metin İşleme ve Embedding Modülü
Türkçe finansal metinler için optimize edilmiş
"""

import logging
import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from sentence_transformers import SentenceTransformer
from langdetect import detect
import nltk
from nltk.tokenize import sent_tokenize
import os

# NLTK data indirme
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

logger = logging.getLogger(__name__)

@dataclass
class TextChunk:
    """Metin parçası container'ı"""
    content: str
    chunk_id: str
    source_file: str
    page_number: int
    chunk_index: int
    content_type: str  # 'text_pdf', 'table_pdf', 'excel', etc.
    embedding: Optional[np.ndarray] = None
    metadata: Dict = None
    language: str = 'tr'
    char_count: int = 0

@dataclass 
class EmbeddingResult:
    """Embedding sonucu"""
    chunks: List[TextChunk]
    total_chunks: int
    embedding_model: str
    processing_time: float

class EmbeddingService:
    """
    Multilingual Sentence Transformers servisi
    Türkçe finansal metinler için optimize edilmiş
    """
    
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        """
        Args:
            model_name: Kullanılacak embedding modeli
        """
        self.model_name = model_name
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Embedding modelini yükle"""
        try:
            logger.info(f"🧠 Embedding modeli yükleniyor: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info("✅ Embedding modeli yüklendi")
        except Exception as e:
            logger.error(f"❌ Embedding modeli yüklenemedi: {e}")
            # Fallback model
            try:
                logger.info("🔄 Fallback model deneniyor...")
                self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
                self.model = SentenceTransformer(self.model_name)
                logger.info("✅ Fallback model yüklendi")
            except Exception as e2:
                logger.error(f"❌ Fallback model de yüklenemedi: {e2}")
                raise
    
    def encode(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        """Metinleri embedding'e çevir"""
        if not self.model:
            raise ValueError("Embedding modeli yüklenmedi")
        
        try:
            embeddings = self.model.encode(
                texts,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            return embeddings
        except Exception as e:
            logger.error(f"❌ Embedding oluşturulamadı: {e}")
            raise
    
    def get_model_info(self) -> Dict:
        """Model bilgilerini döndür"""
        return {
            'model_name': self.model_name,
            'model_loaded': self.model is not None,
            'embedding_dim': self.model.get_sentence_embedding_dimension() if self.model else None,
            'max_seq_length': getattr(self.model, 'max_seq_length', None) if self.model else None
        }

class TextProcessor:
    """
    Metin işleme ve chunking servisi
    PDF ve Excel metinleri için optimize edilmiş
    """
    
    def __init__(self, 
                 chunk_size: int = 800,
                 overlap_size: int = 150,
                 embedding_model: str = None,
                 embedding_service: EmbeddingService = None):
        """
        Args:
            chunk_size: Chunk boyutu (karakter)
            overlap_size: Chunk'lar arası örtüşme
            embedding_model: Embedding model adı (eğer yeni service yaratılacaksa)
            embedding_service: Mevcut embedding service (performance için)
        """
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        
        # Embedding servisini başlat - mevcut service öncelikli
        if embedding_service is not None:
            self.embedding_service = embedding_service
            logger.info(f"📝 TextProcessor başlatıldı - mevcut embedding service kullanılıyor")
        elif embedding_model is not None:
            self.embedding_service = EmbeddingService(embedding_model)
            logger.info(f"📝 TextProcessor başlatıldı - yeni embedding service: {embedding_model}")
        else:
            # Default model
            default_model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            self.embedding_service = EmbeddingService(default_model)
            logger.info(f"📝 TextProcessor başlatıldı - default embedding service: {default_model}")
        
        logger.info(f"📏 Chunk ayarları - boyut: {chunk_size}, overlap: {overlap_size}")
    
    def process_document_pages(self, pages: List, source_file: str) -> List[TextChunk]:
        """
        PDF sayfalarını işle ve chunk'lara böl
        
        Args:
            pages: PageData listesi
            source_file: Kaynak dosya adı
            
        Returns:
            List[TextChunk]: İşlenmiş chunk'lar
        """
        logger.info(f"📄 Doküman işleniyor: {source_file}")
        
        all_chunks = []
        chunk_counter = 0
        
        for page in pages:
            page_number = page.page_number
            
            # Sayfa metnini işle
            if page.text and len(page.text.strip()) > 20:
                text_chunks = self._create_text_chunks(
                    text=page.text,
                    source_file=source_file,
                    page_number=page_number,
                    content_type='text_pdf',
                    start_chunk_id=chunk_counter
                )
                all_chunks.extend(text_chunks)
                chunk_counter += len(text_chunks)
            
            # Tabloları işle
            for table in page.tables:
                if table.content and len(table.content.strip()) > 50:
                    table_chunks = self._create_text_chunks(
                        text=table.content,
                        source_file=source_file,
                        page_number=page_number,
                        content_type='table_pdf',
                        start_chunk_id=chunk_counter
                    )
                    all_chunks.extend(table_chunks)
                    chunk_counter += len(table_chunks)
        
        # Embedding'leri oluştur
        if all_chunks:
            all_chunks = self._create_embeddings(all_chunks)
        
        logger.info(f"✅ {len(all_chunks)} chunk oluşturuldu")
        return all_chunks
    
    def process_excel_sheets(self, sheets: List, source_file: str) -> List[TextChunk]:
        """
        Excel sayfalarını işle ve chunk'lara böl
        
        Args:
            sheets: SheetData listesi
            source_file: Kaynak dosya adı
            
        Returns:
            List[TextChunk]: İşlenmiş chunk'lar
        """
        logger.info(f"📊 Excel işleniyor: {source_file}")
        
        all_chunks = []
        chunk_counter = 0
        
        for sheet_idx, sheet in enumerate(sheets):
            if sheet.text_content and len(sheet.text_content.strip()) > 50:
                excel_chunks = self._create_text_chunks(
                    text=sheet.text_content,
                    source_file=source_file,
                    page_number=sheet_idx + 1,  # Sayfa numarası olarak sheet index kullan
                    content_type='excel',
                    start_chunk_id=chunk_counter
                )
                
                # Excel metadata'sını ekle
                for chunk in excel_chunks:
                    if chunk.metadata is None:
                        chunk.metadata = {}
                    chunk.metadata.update({
                        'sheet_name': sheet.sheet_name,
                        'excel_metadata': sheet.metadata
                    })
                
                all_chunks.extend(excel_chunks)
                chunk_counter += len(excel_chunks)
        
        # Embedding'leri oluştur
        if all_chunks:
            all_chunks = self._create_embeddings(all_chunks)
        
        logger.info(f"✅ Excel {len(all_chunks)} chunk oluşturuldu")
        return all_chunks
    
    def _create_text_chunks(self, 
                           text: str, 
                           source_file: str, 
                           page_number: int,
                           content_type: str,
                           start_chunk_id: int = 0) -> List[TextChunk]:
        """Metni chunk'lara böl"""
        
        if not text or len(text.strip()) < 20:
            return []
        
        # Metni temizle
        clean_text = self._clean_text(text)
        
        # Dil tespiti
        language = self._detect_language(clean_text)
        
        # Chunk'lara böl
        chunks = []
        
        # Eğer metin chunk_size'dan küçükse tek chunk
        if len(clean_text) <= self.chunk_size:
            chunk = TextChunk(
                content=clean_text,
                chunk_id=f"{source_file}_p{page_number}_c{start_chunk_id}",
                source_file=source_file,
                page_number=page_number,
                chunk_index=start_chunk_id,
                content_type=content_type,
                language=language,
                char_count=len(clean_text),
                metadata={'original_length': len(text)}
            )
            chunks.append(chunk)
            return chunks
        
        # Büyük metinleri böl
        sentences = self._split_into_sentences(clean_text, language)
        
        current_chunk = ""
        chunk_index = start_chunk_id
        
        for sentence in sentences:
            # Eğer cümle tek başına chunk_size'dan büyükse, bölmek gerekiyor
            if len(sentence) > self.chunk_size:
                # Önce mevcut chunk'ı kaydet
                if current_chunk.strip():
                    chunk = self._create_chunk(
                        current_chunk.strip(),
                        source_file,
                        page_number,
                        chunk_index,
                        content_type,
                        language,
                        {'original_length': len(text)}
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                    current_chunk = ""
                
                # Uzun cümleyi karakter bazında böl
                sentence_chunks = self._split_long_sentence(sentence)
                for sent_chunk in sentence_chunks:
                    chunk = self._create_chunk(
                        sent_chunk,
                        source_file,
                        page_number,
                        chunk_index,
                        content_type,
                        language,
                        {'original_length': len(text), 'long_sentence_split': True}
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                continue
            
            # Normal chunk birleştirme
            if len(current_chunk) + len(sentence) <= self.chunk_size:
                current_chunk += sentence + " "
            else:
                # Mevcut chunk'ı kaydet
                if current_chunk.strip():
                    chunk = self._create_chunk(
                        current_chunk.strip(),
                        source_file,
                        page_number,
                        chunk_index,
                        content_type,
                        language,
                        {'original_length': len(text)}
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                
                # Yeni chunk'ı başlat (overlap ile)
                current_chunk = self._create_overlap(current_chunk, sentence)
        
        # Son chunk'ı kaydet
        if current_chunk.strip():
            chunk = self._create_chunk(
                current_chunk.strip(),
                source_file,
                page_number,
                chunk_index,
                content_type,
                language,
                {'original_length': len(text)}
            )
            chunks.append(chunk)
        
        return chunks
    
    def _create_chunk(self, content: str, source_file: str, page_number: int, 
                     chunk_index: int, content_type: str, language: str, metadata: Dict) -> TextChunk:
        """Tek bir chunk oluştur"""
        return TextChunk(
            content=content,
            chunk_id=f"{source_file}_p{page_number}_c{chunk_index}",
            source_file=source_file,
            page_number=page_number,
            chunk_index=chunk_index,
            content_type=content_type,
            language=language,
            char_count=len(content),
            metadata=metadata
        )
    
    def _create_overlap(self, current_chunk: str, new_sentence: str) -> str:
        """Overlap oluştur"""
        if len(current_chunk) <= self.overlap_size:
            return new_sentence + " "
        
        # Son overlap_size kadar karakteri al
        overlap_text = current_chunk[-self.overlap_size:]
        
        # Kelime sınırında kes
        words = overlap_text.split()
        if len(words) > 1:
            overlap_text = " ".join(words[1:])  # İlk kısmi kelimeyi at
        
        return overlap_text + " " + new_sentence + " "
    
    def _split_long_sentence(self, sentence: str) -> List[str]:
        """Uzun cümleyi böl"""
        chunks = []
        current = ""
        
        words = sentence.split()
        for word in words:
            if len(current) + len(word) + 1 <= self.chunk_size:
                current += word + " "
            else:
                if current.strip():
                    chunks.append(current.strip())
                current = word + " "
        
        if current.strip():
            chunks.append(current.strip())
        
        return chunks
    
    def _split_into_sentences(self, text: str, language: str) -> List[str]:
        """Metni cümlelere böl"""
        try:
            # NLTK ile cümlelere böl
            if language == 'tr':
                # Türkçe için özel işleme
                sentences = sent_tokenize(text, language='turkish')
            else:
                sentences = sent_tokenize(text)
            
            # Çok kısa cümleleri birleştir
            merged_sentences = []
            current = ""
            
            for sentence in sentences:
                if len(sentence) < 50 and current:
                    current += " " + sentence
                else:
                    if current:
                        merged_sentences.append(current)
                    current = sentence
            
            if current:
                merged_sentences.append(current)
            
            return merged_sentences if merged_sentences else [text]
        
        except Exception as e:
            logger.warning(f"⚠️ Cümle bölme hatası: {e}")
            # Fallback: nokta ile böl
            sentences = re.split(r'[.!?]+', text)
            return [s.strip() for s in sentences if s.strip()]
    
    def _clean_text(self, text: str) -> str:
        """Metni temizle"""
        if not text:
            return ""
        
        # Fazla boşlukları kaldır
        text = re.sub(r'\s+', ' ', text)
        
        # Özel karakterleri temizle
        text = re.sub(r'[^\w\s.,!?;:-]', ' ', text)
        
        # Fazla noktalama işaretlerini temizle
        text = re.sub(r'\.{3,}', '...', text)
        text = re.sub(r'-{2,}', '--', text)
        
        return text.strip()
    
    def _detect_language(self, text: str) -> str:
        """Dil tespiti"""
        try:
            if len(text) > 50:
                lang = detect(text)
                return lang
        except:
            pass
        return 'tr'  # Varsayılan Türkçe
    
    def _create_embeddings(self, chunks: List[TextChunk], batch_size: int = 32) -> List[TextChunk]:
        """
        Chunk'lar için embedding oluştur (batch processing ile optimize edilmiş)
        
        Args:
            chunks: İşlenecek chunk'lar
            batch_size: Batch boyutu (GPU memory'ye göre ayarlanabilir)
        """
        if not chunks:
            return chunks
        
        logger.info(f"🧠 {len(chunks)} chunk için embedding oluşturuluyor... (batch_size: {batch_size})")
        
        try:
            import time
            start_time = time.time()
            
            # Metinleri çıkar
            texts = [chunk.content for chunk in chunks]
            
            # Büyük chunk setleri için batch processing
            if len(texts) > batch_size:
                logger.info(f"📦 Büyük chunk seti - batch processing kullanılıyor")
                
                all_embeddings = []
                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i:i + batch_size]
                    batch_embeddings = self.embedding_service.encode(
                        batch_texts, 
                        show_progress=True
                    )
                    all_embeddings.extend(batch_embeddings)
                    
                    # Progress logging
                    processed = min(i + batch_size, len(texts))
                    logger.debug(f"🔄 {processed}/{len(texts)} chunks işlendi")
                
                embeddings = np.array(all_embeddings)
            else:
                # Küçük chunk setleri için tek seferde
                embeddings = self.embedding_service.encode(texts, show_progress=True)
            
            # Chunk'lara embedding'leri ekle
            for i, chunk in enumerate(chunks):
                chunk.embedding = embeddings[i]
            
            # Performance metrics
            processing_time = time.time() - start_time
            chunks_per_second = len(chunks) / processing_time
            
            logger.info(f"✅ Embedding'ler oluşturuldu ({processing_time:.2f}s, {chunks_per_second:.1f} chunks/s)")
            
        except Exception as e:
            logger.error(f"❌ Embedding oluşturulamadı: {e}")
            # Embedding olmadan devam et
        
        return chunks
    
    def embed_chunks(self, chunks: List[TextChunk], batch_size: int = None) -> List[TextChunk]:
        """
        Public method for embedding chunks with automatic batch size optimization
        
        Args:
            chunks: İşlenecek chunk'lar
            batch_size: Manuel batch size (None ise otomatik)
        """
        if not chunks:
            return chunks
        
        # Otomatik batch size optimization
        if batch_size is None:
            # GPU memory ve chunk sayısına göre optimize et
            chunk_count = len(chunks)
            if chunk_count <= 16:
                batch_size = chunk_count  # Küçük setler için tek batch
            elif chunk_count <= 100:
                batch_size = 32  # Orta setler için 32
            elif chunk_count <= 500:
                batch_size = 64  # Büyük setler için 64
            else:
                batch_size = 128  # Çok büyük setler için 128
            
            logger.debug(f"🎯 Otomatik batch size: {batch_size} (toplam chunks: {chunk_count})")
        
        return self._create_embeddings(chunks, batch_size)
    
    def create_chunks(self, text: str, source_file: str, page_number: int, 
                     content_type: str, metadata: Dict = None) -> List[TextChunk]:
        """
        Public method for creating chunks from text
        
        Args:
            text: İşlenecek metin
            source_file: Kaynak dosya adı
            page_number: Sayfa numarası
            content_type: İçerik türü
            metadata: Ek metadata
        """
        chunks = self._create_text_chunks(
            text=text,
            source_file=source_file,
            page_number=page_number,
            content_type=content_type
        )
        
        # Metadata ekle
        if metadata:
            for chunk in chunks:
                if chunk.metadata is None:
                    chunk.metadata = {}
                chunk.metadata.update(metadata)
        
        return chunks
    
    def get_processing_stats(self, chunks: List[TextChunk] = None) -> Dict:
        """İşleme istatistiklerini döndür"""
        if not chunks:
            return {
                'total_chunks': 0,
                'text_chunks': 0,
                'table_chunks': 0,
                'excel_chunks': 0,
                'total_characters': 0,
                'avg_chunk_size': 0,
                'language_distribution': {},
                'content_type_distribution': {},
                'embedding_model': self.embedding_service.model_name
            }
        
        stats = {
            'total_chunks': len(chunks),
            'total_characters': sum(chunk.char_count for chunk in chunks),
            'embedding_model': self.embedding_service.model_name
        }
        
        # İçerik türü dağılımı
        content_types = {}
        for chunk in chunks:
            content_types[chunk.content_type] = content_types.get(chunk.content_type, 0) + 1
        
        stats['content_type_distribution'] = content_types
        stats['text_chunks'] = content_types.get('text_pdf', 0)
        stats['table_chunks'] = content_types.get('table_pdf', 0) + content_types.get('table_pdf_fragment', 0)
        stats['excel_chunks'] = content_types.get('excel', 0)
        
        # Dil dağılımı
        languages = {}
        for chunk in chunks:
            languages[chunk.language] = languages.get(chunk.language, 0) + 1
        stats['language_distribution'] = languages
        
        # Ortalama chunk boyutu
        if chunks:
            stats['avg_chunk_size'] = stats['total_characters'] / len(chunks)
            
            # Tablo sayısı hesaplama
            table_pages = set()
            for chunk in chunks:
                if chunk.content_type in ['table_pdf', 'table_pdf_fragment']:
                    table_pages.add(f"{chunk.source_file}_p{chunk.page_number}")
            stats['pages_with_tables'] = len(table_pages)
        
        return stats 