"""
PDF İşleme Modülü
Finansal PDF dokümanlarını okuma ve tablo çıkarma için optimize edilmiş
"""

import logging
import fitz  # PyMuPDF
import pdfplumber
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

logger = logging.getLogger(__name__)

@dataclass
class TableData:
    """Tablo verisi container'ı"""
    content: str
    page_number: int
    table_index: int
    rows: int
    columns: int
    metadata: Dict

@dataclass
class PageData:
    """Sayfa verisi container'ı"""
    text: str
    tables: List[TableData]
    page_number: int
    metadata: Dict

@dataclass
class PDFResult:
    """PDF işleme sonucu"""
    pages: List[PageData]
    total_pages: int
    total_tables: int
    metadata: Dict

class PDFProcessor:
    """
    Finansal PDF dokümanları için optimize edilmiş işlemci
    Multiprocessing ve gelişmiş tablo çıkarma destekli
    """
    
    def __init__(self, max_workers: int = None):
        """
        Args:
            max_workers: Paralel işlem için maksimum worker sayısı
        """
        self.max_workers = max_workers or min(4, os.cpu_count())
        logger.info(f"📄 PDFProcessor başlatıldı - {self.max_workers} worker")
    
    def process_pdf(self, pdf_path: str) -> PDFResult:
        """
        PDF dosyasını işle ve sonuçları döndür
        
        Args:
            pdf_path: PDF dosyasının yolu
            
        Returns:
            PDFResult: İşlenmiş PDF verisi
        """
        logger.info(f"📄 PDF işleniyor: {pdf_path}")
        
        try:
            # PDF metadata'sını al
            pdf_metadata = self._get_pdf_metadata(pdf_path)
            total_pages = pdf_metadata['page_count']
            
            logger.info(f"📊 {total_pages} sayfa bulundu")
            
            # Sayfaları paralel işle
            pages = self._process_pages_parallel(pdf_path, total_pages)
            
            # İstatistikleri hesapla
            total_tables = sum(len(page.tables) for page in pages)
            
            result = PDFResult(
                pages=pages,
                total_pages=total_pages,
                total_tables=total_tables,
                metadata=pdf_metadata
            )
            
            logger.info(f"✅ PDF işlendi - {total_pages} sayfa, {total_tables} tablo")
            return result
            
        except Exception as e:
            logger.error(f"❌ PDF işleme hatası: {e}")
            raise
    
    def _get_pdf_metadata(self, pdf_path: str) -> Dict:
        """PDF metadata'sını çıkar"""
        try:
            with fitz.open(pdf_path) as doc:
                metadata = doc.metadata or {}
                return {
                    'page_count': len(doc),
                    'title': metadata.get('title', ''),
                    'author': metadata.get('author', ''),
                    'subject': metadata.get('subject', ''),
                    'creator': metadata.get('creator', ''),
                    'producer': metadata.get('producer', ''),
                    'creation_date': metadata.get('creationDate', ''),
                    'modification_date': metadata.get('modDate', '')
                }
        except Exception as e:
            logger.warning(f"⚠️ Metadata alınamadı: {e}")
            return {'page_count': 0}
    
    def _process_pages_parallel(self, pdf_path: str, total_pages: int) -> List[PageData]:
        """Sayfaları paralel olarak işle"""
        pages = []
        
        # Küçük ve orta PDF'ler için sıralı işleme (daha hızlı)
        if total_pages <= 10:  # 10 sayfaya kadar sıralı işleme
            logger.info(f"📄 {total_pages} sayfa - optimize edilmiş sıralı işleme")
            return self._process_pages_sequential_optimized(pdf_path, total_pages)
        
        # Büyük PDF'ler için paralel işleme
        logger.info(f"🚀 Paralel işleme başlatılıyor - {self.max_workers} worker")
        
        # Mac için multiprocessing problemlerini azalt
        import platform
        if platform.system() == 'Darwin':  # macOS
            logger.info("🍎 macOS tespit edildi - sıralı işleme kullanılıyor")
            return self._process_pages_sequential_optimized(pdf_path, total_pages)
        
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Tüm sayfalar için task'ları başlat
            future_to_page = {
                executor.submit(self._process_single_page, pdf_path, page_num): page_num
                for page_num in range(total_pages)
            }
            
            # Sonuçları topla
            for future in as_completed(future_to_page):
                page_num = future_to_page[future]
                try:
                    page_data = future.result()
                    if page_data:
                        pages.append(page_data)
                        logger.debug(f"✅ Sayfa {page_num + 1} işlendi")
                except Exception as e:
                    logger.error(f"❌ Sayfa {page_num + 1} işlenemedi: {e}")
        
        # Sayfa sırasına göre sırala
        pages.sort(key=lambda x: x.page_number)
        logger.info(f"✅ {len(pages)} sayfa başarıyla işlendi")
        return pages
    
    def _process_pages_sequential_optimized(self, pdf_path: str, total_pages: int) -> List[PageData]:
        """
        Optimize edilmiş sıralı sayfa işleme
        PDF'i bir kez açıp tüm sayfaları işler
        """
        pages = []
        
        try:
            # PyMuPDF ile bir kez aç
            with fitz.open(pdf_path) as doc:
                logger.info(f"📖 PDF açıldı - {total_pages} sayfa işlenecek")
                
                # pdfplumber ile de bir kez aç
                import pdfplumber
                with pdfplumber.open(pdf_path) as pdf_plumber:
                    
                    for page_num in range(total_pages):
                        # Progress göster
                        if page_num % 5 == 0 or page_num == total_pages - 1:
                            logger.info(f"📄 Sayfa {page_num + 1}/{total_pages} işleniyor...")
                        
                        # Metin çıkarma (PyMuPDF - zaten açık)
                        text = self._extract_text_from_open_doc(doc, page_num)
                        
                        # Tablo çıkarma (pdfplumber - zaten açık)
                        tables = self._extract_tables_from_open_pdf(pdf_plumber, page_num)
                        
                        # Sayfa metadata'sı
                        metadata = {
                            'text_length': len(text),
                            'table_count': len(tables),
                            'has_text': len(text.strip()) > 10,
                            'has_tables': len(tables) > 0
                        }
                        
                        page_data = PageData(
                            text=text,
                            tables=tables,
                            page_number=page_num + 1,  # 1-indexed
                            metadata=metadata
                        )
                        
                        pages.append(page_data)
            
            logger.info(f"✅ {len(pages)} sayfa optimize edilmiş şekilde işlendi")
            return pages
            
        except Exception as e:
            logger.error(f"❌ Optimize edilmiş sayfa işleme hatası: {e}")
            # Fallback to original method
            return self._process_pages_fallback(pdf_path, total_pages)
    
    def _extract_text_from_open_doc(self, doc, page_num: int) -> str:
        """Açık PyMuPDF dökümanından metin çıkar"""
        try:
            page = doc[page_num]
            text = page.get_text()
            return self._clean_text(text)
        except Exception as e:
            logger.warning(f"⚠️ PyMuPDF metin çıkarma hatası sayfa {page_num + 1}: {e}")
            return ""
    
    def _extract_tables_from_open_pdf(self, pdf_plumber, page_num: int) -> List[TableData]:
        """Açık pdfplumber dökümanından tablo çıkar"""
        tables = []
        
        try:
            if page_num >= len(pdf_plumber.pages):
                return tables
            
            page = pdf_plumber.pages[page_num]
            extracted_tables = page.extract_tables()
            
            if not extracted_tables:
                return tables
            
            for table_idx, table in enumerate(extracted_tables):
                if not table or len(table) < 2:  # Çok küçük tablolar
                    continue
                
                # Optimize edilmiş tablo formatlama
                formatted_table = self._format_table_fast(table)
                
                if formatted_table and len(formatted_table) > 20:
                    table_data = TableData(
                        content=formatted_table,
                        page_number=page_num + 1,
                        table_index=table_idx,
                        rows=len(table),
                        columns=len(table[0]) if table else 0,
                        metadata={
                            'extraction_method': 'pdfplumber_optimized',
                            'table_area': 'detected'
                        }
                    )
                    tables.append(table_data)
            
            return tables
            
        except Exception as e:
            logger.warning(f"⚠️ Tablo çıkarma hatası sayfa {page_num + 1}: {e}")
            return tables
    
    def _format_table_fast(self, table: List[List]) -> str:
        """Hızlı tablo formatlama (DataFrame olmadan)"""
        if not table:
            return ""
        
        try:
            formatted = "TABLO:\n"
            
            # Maksimum 50 satır işle (performance için)
            max_rows = min(50, len(table))
            
            for i, row in enumerate(table[:max_rows]):
                if not row:
                    continue
                
                # None değerleri temizle
                cleaned_row = [str(cell).strip() if cell is not None else "" for cell in row]
                
                # Boş satırları atla
                if not any(cell for cell in cleaned_row):
                    continue
                
                # Satırı formatla
                row_text = " | ".join(cleaned_row[:10])  # Maksimum 10 sütun
                if row_text.strip():
                    formatted += row_text + "\n"
                
                # Performance: çok uzun tablolar için break
                if i > 30:  # Maksimum 30 satır
                    formatted += "... (tablo devam ediyor)\n"
                    break
            
            return formatted
            
        except Exception as e:
            logger.warning(f"⚠️ Hızlı tablo formatlama hatası: {e}")
            return "TABLO: (formatlanamadı)\n"
    
    def _process_pages_fallback(self, pdf_path: str, total_pages: int) -> List[PageData]:
        """Fallback: eski yöntem"""
        pages = []
        logger.info("🔄 Fallback işleme moduna geçiliyor")
        
        for page_num in range(total_pages):
            page_data = self._process_single_page(pdf_path, page_num)
            if page_data:
                pages.append(page_data)
                
        return pages
    
    def _process_single_page(self, pdf_path: str, page_num: int) -> Optional[PageData]:
        """Tek bir sayfayı işle"""
        try:
            # Metin çıkarma (PyMuPDF)
            text = self._extract_text_pymupdf(pdf_path, page_num)
            
            # Tablo çıkarma (pdfplumber)
            tables = self._extract_tables_pdfplumber(pdf_path, page_num)
            
            # Sayfa metadata'sı
            metadata = {
                'text_length': len(text),
                'table_count': len(tables),
                'has_text': len(text.strip()) > 10,
                'has_tables': len(tables) > 0
            }
            
            return PageData(
                text=text,
                tables=tables,
                page_number=page_num + 1,  # 1-indexed
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"❌ Sayfa {page_num + 1} işlenemedi: {e}")
            return None
    
    def _extract_text_pymupdf(self, pdf_path: str, page_num: int) -> str:
        """PyMuPDF ile metin çıkar"""
        try:
            with fitz.open(pdf_path) as doc:
                page = doc[page_num]
                text = page.get_text()
                
                # Temizleme
                text = self._clean_text(text)
                return text
                
        except Exception as e:
            logger.warning(f"⚠️ PyMuPDF metin çıkarma hatası sayfa {page_num + 1}: {e}")
            return ""
    
    def _extract_tables_pdfplumber(self, pdf_path: str, page_num: int) -> List[TableData]:
        """pdfplumber ile tablo çıkar"""
        tables = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                if page_num >= len(pdf.pages):
                    return tables
                
                page = pdf.pages[page_num]
                extracted_tables = page.extract_tables()
                
                if not extracted_tables:
                    return tables
                
                for table_idx, table in enumerate(extracted_tables):
                    if not table or len(table) < 2:  # Çok küçük tablolar
                        continue
                    
                    # Tabloyu temizle ve formatla
                    formatted_table = self._format_table(table)
                    
                    if formatted_table and len(formatted_table) > 20:  # Minimum içerik kontrolü
                        table_data = TableData(
                            content=formatted_table,
                            page_number=page_num + 1,
                            table_index=table_idx,
                            rows=len(table),
                            columns=len(table[0]) if table else 0,
                            metadata={
                                'extraction_method': 'pdfplumber',
                                'table_area': 'detected'
                            }
                        )
                        tables.append(table_data)
                
                logger.debug(f"📊 Sayfa {page_num + 1}: {len(tables)} tablo bulundu")
                
        except Exception as e:
            logger.warning(f"⚠️ Tablo çıkarma hatası sayfa {page_num + 1}: {e}")
        
        return tables
    
    def _format_table(self, table: List[List]) -> str:
        """Tabloyu okunabilir metin formatına çevir"""
        if not table:
            return ""
        
        try:
            # None değerleri temizle
            cleaned_table = []
            for row in table:
                cleaned_row = [str(cell).strip() if cell is not None else "" for cell in row]
                # Boş satırları atla
                if any(cell for cell in cleaned_row):
                    cleaned_table.append(cleaned_row)
            
            if not cleaned_table:
                return ""
            
            # DataFrame'e çevir
            df = pd.DataFrame(cleaned_table)
            
            # Boş sütunları kaldır
            df = df.dropna(axis=1, how='all')
            df = df.replace('', np.nan).dropna(axis=1, how='all')
            
            if df.empty:
                return ""
            
            # İlk satırı header olarak kullan
            if len(df) > 1:
                df.columns = df.iloc[0]
                df = df.drop(df.index[0]).reset_index(drop=True)
            
            # Metin formatına çevir
            formatted_text = "TABLO:\n"
            formatted_text += df.to_string(index=False, na_rep='')
            formatted_text += "\n"
            
            return formatted_text
            
        except Exception as e:
            logger.warning(f"⚠️ Tablo formatlama hatası: {e}")
            # Fallback: Basit formatla
            try:
                formatted = "TABLO:\n"
                for row in table[:10]:  # İlk 10 satır
                    if row:
                        row_text = " | ".join(str(cell) if cell else "" for cell in row)
                        if row_text.strip():
                            formatted += row_text + "\n"
                return formatted
            except:
                return ""
    
    def _clean_text(self, text: str) -> str:
        """Metni temizle"""
        if not text:
            return ""
        
        # Fazla boşlukları kaldır
        text = ' '.join(text.split())
        
        # Çok kısa metinleri filtrele
        if len(text.strip()) < 10:
            return ""
        
        return text.strip()
    
    def get_processor_stats(self) -> Dict:
        """İşlemci istatistiklerini döndür"""
        return {
            'processor_type': 'PDFProcessor',
            'max_workers': self.max_workers,
            'supported_formats': ['pdf'],
            'features': [
                'parallel_processing',
                'table_extraction',
                'text_extraction',
                'metadata_extraction'
            ]
        }

# Utility functions
def process_single_page_worker(args: Tuple[str, int]) -> Optional[PageData]:
    """Worker function for multiprocessing"""
    pdf_path, page_num = args
    processor = PDFProcessor(max_workers=1)
    return processor._process_single_page(pdf_path, page_num) 