"""
Test Query Generator for Turkish Financial RAG System
Generates 20-30 test queries for manual accuracy evaluation
"""

from evaluation.metrics.accuracy_tracker import AccuracyTracker, QueryCategory, get_global_accuracy_tracker

def generate_test_queries():
    """Generate comprehensive test queries for manual evaluation"""
    
    tracker = get_global_accuracy_tracker()
    
    # Test queries organized by category
    test_queries = [
        # BALANCE_SHEET queries (Bilanço)
        {
            "query": "Şirketin toplam varlıkları ne kadar?",
            "category": QueryCategory.BALANCE_SHEET,
            "expected": "Finansal tablolarda toplam aktif değeri"
        },
        {
            "query": "Şirketin öz kaynak tutarı nedir?",
            "category": QueryCategory.BALANCE_SHEET,
            "expected": "Toplam özkaynak miktarı"
        },
        {
            "query": "Nakit ve nakit benzeri varlıklar ne kadar?",
            "category": QueryCategory.BALANCE_SHEET,
            "expected": "Nakit varlıklar toplamı"
        },
        {
            "query": "Şirketin toplam borçları ne kadar?",
            "category": QueryCategory.BALANCE_SHEET,
            "expected": "Kısa vadeli + uzun vadeli borçlar toplamı"
        },
        
        # INCOME_STATEMENT queries (Gelir Tablosu)
        {
            "query": "Şirketin net satış hasılatı ne kadar?",
            "category": QueryCategory.INCOME_STATEMENT,
            "expected": "Dönem net satış hasılatı"
        },
        {
            "query": "Net kar ne kadar gerçekleşti?",
            "category": QueryCategory.INCOME_STATEMENT,
            "expected": "Dönem net karı"
        },
        {
            "query": "Faaliyet karı nedir?",
            "category": QueryCategory.INCOME_STATEMENT,
            "expected": "Esas faaliyet karı"
        },
        {
            "query": "Finansman giderleri ne kadar?",
            "category": QueryCategory.INCOME_STATEMENT,
            "expected": "Toplam finansman giderleri"
        },
        {
            "query": "Satışların maliyeti ne kadar?",
            "category": QueryCategory.INCOME_STATEMENT,
            "expected": "Satılan malların maliyeti"
        },
        
        # CASH_FLOW queries (Nakit Akışı)
        {
            "query": "İşletme faaliyetlerinden nakit akışı nedir?",
            "category": QueryCategory.CASH_FLOW,
            "expected": "Operasyonel nakit akışı"
        },
        {
            "query": "Yatırım faaliyetlerinden nakit akışı ne kadar?",
            "category": QueryCategory.CASH_FLOW,
            "expected": "Yatırım nakit akışı"
        },
        {
            "query": "Finansman faaliyetlerinden nakit akışı nedir?",
            "category": QueryCategory.CASH_FLOW,
            "expected": "Finansman nakit akışı"
        },
        
        # RATIO_ANALYSIS queries (Oran Analizi)
        {
            "query": "Şirketin cari oranı nedir?",
            "category": QueryCategory.RATIO_ANALYSIS,
            "expected": "Cari varlıklar / Kısa vadeli borçlar"
        },
        {
            "query": "Borç/özkaynak oranı ne kadar?",
            "category": QueryCategory.RATIO_ANALYSIS,
            "expected": "Toplam borç / Özkaynak oranı"
        },
        {
            "query": "Net kar marjı nedir?",
            "category": QueryCategory.RATIO_ANALYSIS,
            "expected": "Net kar / Satış hasılatı * 100"
        },
        {
            "query": "Özkaynak kârlılığı (ROE) nedir?",
            "category": QueryCategory.RATIO_ANALYSIS,
            "expected": "Net kar / Özkaynak * 100"
        },
        {
            "query": "Aktif kârlılığı (ROA) ne kadar?",
            "category": QueryCategory.RATIO_ANALYSIS,
            "expected": "Net kar / Toplam aktif * 100"
        },
        
        # GENERAL_INFO queries (Genel Bilgi)
        {
            "query": "Şirketin faaliyet alanı nedir?",
            "category": QueryCategory.GENERAL_INFO,
            "expected": "Ana faaliyet konusu"
        },
        {
            "query": "Şirket hangi sektörde faaliyet gösteriyor?",
            "category": QueryCategory.GENERAL_INFO,
            "expected": "Sektör bilgisi"
        },
        {
            "query": "Şirketin kuruluş tarihi nedir?",
            "category": QueryCategory.GENERAL_INFO,
            "expected": "Kuruluş yılı"
        },
        {
            "query": "Çalışan sayısı ne kadar?",
            "category": QueryCategory.GENERAL_INFO,
            "expected": "Toplam personel sayısı"
        },
        
        # COMPARISON queries (Karşılaştırma)
        {
            "query": "Bu yılki satışlar geçen yıla göre nasıl değişti?",
            "category": QueryCategory.COMPARISON,
            "expected": "Yıllık satış değişimi ve oranı"
        },
        {
            "query": "Net kar geçen yıla göre arttı mı azaldı mı?",
            "category": QueryCategory.COMPARISON,
            "expected": "Net kar değişim yönü ve miktarı"
        },
        {
            "query": "Şirketin borç seviyesi önceki döneme göre nasıl?",
            "category": QueryCategory.COMPARISON,
            "expected": "Borç artış/azalış oranı"
        },
        
        # TREND_ANALYSIS queries (Trend Analizi)
        {
            "query": "Son 3 yılda satış trendi nasıl?",
            "category": QueryCategory.TREND_ANALYSIS,
            "expected": "3 yıllık satış trend analizi"
        },
        {
            "query": "Kârlılık trendleri nasıl seyrediyor?",
            "category": QueryCategory.TREND_ANALYSIS,
            "expected": "Kar marjı trend analizi"
        },
        {
            "query": "Şirketin büyüme hızı nedir?",
            "category": QueryCategory.TREND_ANALYSIS,
            "expected": "Yıllık büyüme oranları"
        },
        
        # Complex queries (Karmaşık Sorular)
        {
            "query": "Şirketin finansal durumu genel olarak nasıl değerlendiriliyor?",
            "category": QueryCategory.GENERAL_INFO,
            "expected": "Genel finansal durum analizi"
        },
        {
            "query": "Hangi riskler şirketi etkiliyor?",
            "category": QueryCategory.GENERAL_INFO,
            "expected": "Risk faktörleri listesi"
        },
        {
            "query": "Gelecek dönem tahminleri neler?",
            "category": QueryCategory.TREND_ANALYSIS,
            "expected": "Gelecek dönem projeksiyonları"
        },
        {
            "query": "Şirketin rekabet avantajları nelerdir?",
            "category": QueryCategory.GENERAL_INFO,
            "expected": "Rekabetçi üstünlükler"
        }
    ]
    
    # Add all test queries to the tracker
    print("📝 Generating test queries for manual evaluation...")
    print(f"📊 Total queries to add: {len(test_queries)}")
    
    for i, query_data in enumerate(test_queries, 1):
        tracker.add_test_query(
            query=query_data["query"],
            category=query_data["category"],
            expected_answer=query_data["expected"]
        )
        print(f"   ✅ Added query {i:02d}: {query_data['query'][:50]}...")
    
    print(f"\n🎯 Successfully generated {len(test_queries)} test queries!")
    print("📁 Queries saved to: evaluation/data/test_queries.json")
    print("\n📋 Next steps for manual evaluation:")
    print("1. Run these queries through your RAG system")
    print("2. Manually evaluate each response (1-5 scale)")
    print("3. Record evaluations using AccuracyTracker")
    print("4. Generate accuracy report")
    
    # Show category breakdown
    category_counts = {}
    for query_data in test_queries:
        cat = query_data["category"].value
        category_counts[cat] = category_counts.get(cat, 0) + 1
    
    print(f"\n📂 Query distribution by category:")
    for category, count in category_counts.items():
        print(f"   {category.replace('_', ' ').title()}: {count} queries")
    
    return test_queries

def print_evaluation_template():
    """Print a template for manual evaluation"""
    
    template = """
# 📋 Manual Evaluation Template

## Quality Scoring Guide:
- **5 (EXCELLENT)**: Mükemmel - Tam doğru ve kapsamlı cevap
- **4 (GOOD)**: İyi - Doğru ama eksik detay olabilir
- **3 (FAIR)**: Orta - Kısmen doğru cevap
- **2 (POOR)**: Zayıf - Büyük ölçüde yanlış
- **1 (FAILED)**: Başarısız - Tamamen yanlış veya cevap yok

## Evaluation Process:
1. Run query through your RAG system
2. Compare response with expected answer
3. Score 1-5 based on accuracy and completeness
4. Add manual notes if needed
5. Record in AccuracyTracker

## Example Code:
```python
from evaluation.metrics.accuracy_tracker import get_global_accuracy_tracker, ResponseQuality, QueryCategory

tracker = get_global_accuracy_tracker()

# Evaluate a response
tracker.evaluate_response(
    query_id="q_001",
    query_text="Şirketin toplam varlıkları ne kadar?",
    actual_response="Şirketin toplam varlıkları 1.2 milyar TL'dir.",
    quality_score=ResponseQuality.GOOD,  # 4 points
    category=QueryCategory.BALANCE_SHEET,
    response_time=2.3,
    expected_answer="Finansal tablolarda toplam aktif değeri",
    manual_notes="Doğru rakam verdi ama detay eksik"
)
```

## Quick Evaluation:
```python
# For rapid testing
tracker.quick_evaluate(
    query="Test query", 
    response="System response", 
    score=4,  # 1-5 scale
    response_time=1.5
)
```
"""
    
    print(template)
    
    # Save template to file
    with open("evaluation/manual_evaluation_template.md", "w", encoding="utf-8") as f:
        f.write(template)
    
    print("📄 Template saved to: evaluation/manual_evaluation_template.md")

if __name__ == "__main__":
    # Generate test queries
    queries = generate_test_queries()
    
    # Print evaluation template
    print_evaluation_template()
    
    print(f"\n🚀 Ready for manual evaluation of {len(queries)} queries!")
    print("📖 Check the template file for evaluation guidelines.") 