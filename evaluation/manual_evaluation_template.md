
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
