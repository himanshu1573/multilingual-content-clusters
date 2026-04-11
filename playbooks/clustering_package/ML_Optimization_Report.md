# ML Optimization Report: Hinglish Video Title Clustering

## 1. Executive Summary

This document analyzes the performance of our BERTopic-based clustering system for **8,930 multilingual (Hindi+English) video titles** from news channels. The system groups titles into semantic topics and generates search-ready labels using a local LLM.

**Current Performance:**
| Metric | Value | Target |
|--------|-------|--------|
| Documents Processed | 8,930 | - |
| Topics Discovered | 42 | 15-20 |
| Noise Ratio | 25.7% | < 12% |
| Silhouette Score | 0.0596 | > 0.40 |
| Pipeline Runtime | ~4 min | < 5 min |

## 2. Problem Statement

### The Data Challenge
- **Mixed Scripts**: ~70% Hindi (Devanagari) + 30% English (Latin) in the same title
- **Noise**: Hashtags (#shorts, #viral), channel codes (at2, n18v), mentions (@user)
- **Entity Variants**: Same person appears as "Modi", "PM Modi", "Narendra Modi", "मोदी"
- **Short Titles**: Many titles are < 10 words, giving embeddings very little signal

### The Clustering Challenge
- HDBSCAN requires dense clusters. Hinglish titles with mixed scripts create sparse embeddings.
- Titles about the same event (e.g., "Iran war") use different phrasing and get scattered.
- Result: 25.7% of titles (2,295 docs) are marked as noise (-1) even though they are clearly relevant.

## 3. Architecture

```
Input (Video Titles.txt, 9,722 rows)
    │
    ▼
┌─────────────────────────┐
│  Step 1: Data Cleaning  │  Remove URLs, hashtags, mentions, channel codes
│  Step 2: Hindi Preproc  │  80+ Hindi stopwords, preserve short Hindi tokens
│  Step 3: NER Normalize  │  Entity canonicalization (20+ patterns)
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│  Step 4: LaBSE Embed    │  768-dim multilingual sentence embeddings
│  Step 5: UMAP Reduce    │  768 → 5 dimensions, cosine metric
│  Step 6: HDBSCAN        │  Density-based clustering, min_cluster_size=15
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│  Step 7: LLM Labeling   │  Ollama/Qwen2.5:3b generates topic names
│  Step 8: Export CSV      │  5 columns: raw, clean, ner, topic_id, label
└─────────────────────────┘
```

## 4. Key Optimizations Applied

### 4.1 Hindi Stopword Expansion
- **Before**: 10 stopwords (में, के, की, का, पर, ने, को, से, है, क्या)
- **After**: 80+ stopwords covering common verbs, pronouns, postpositions
- **Impact**: Cleaner embeddings, less noise from function words

### 4.2 Token Length Filter Fix
- **Before**: All tokens ≤ 2 chars were removed (broke Hindi words like "हो", "तो")
- **After**: Length filter applies ONLY to ASCII tokens. Hindi tokens filtered by stopword list only.
- **Impact**: Preserved meaningful short Hindi words

### 4.3 Entity Canonicalization
- **Before**: "PM Modi", "modi ji", "Narendra Modi" = 3 different entities
- **After**: All mapped to "pm_modi" = 1 entity
- **Impact**: Reduced cluster fragmentation for person-centric topics

### 4.4 Output Simplification
- **Before**: 10+ columns including source_row, source_id, model_text, coarse_bucket, is_noise
- **After**: 5 columns: raw_text, clean_text, ner_text, topic_id, topic_label
- **Impact**: Clean, readable output for downstream consumption

## 5. Top 10 Discovered Topics

| Rank | Topic | Documents | % of Total |
|------|-------|-----------|-----------|
| 1 | Iran-Israel War | 3,464 | 38.8% |
| 2 | Elections 2026 | 1,268 | 14.2% |
| 3 | IIT Baba Marriage | 513 | 5.7% |
| 4 | Assembly Security Breach | 194 | 2.2% |
| 5 | LPG Crisis | 109 | 1.2% |
| 6 | Yogi Adityanath Activism | 77 | 0.9% |
| 7 | Croc Threatens Home | 72 | 0.8% |
| 8 | Artemis Moon Mission | 63 | 0.7% |
| 9 | CCTV Crime | 53 | 0.6% |
| 10 | Legal Proceedings | 52 | 0.6% |

## 6. Known Issue: High Noise Ratio

**Problem**: 25.7% noise is unacceptable. Many clearly relevant titles are marked -1.

**Root Cause**: HDBSCAN creates clusters only in dense embedding regions. Titles with unique phrasing or mixed script combinations fall in sparse areas.

**Example Noise Titles** (should NOT be noise):
- "trump iran फाइनल वार्निंग" → Should be Iran-Israel War
- "bengal elections 2026 sir कटे करीब लाख नाम" → Should be Elections 2026
- "iran war saudi arabia रक्षा मंत्रालय" → Should be Iran-Israel War

**Planned Fix**: Outlier reduction via cosine similarity re-assignment. Each noise document gets assigned to its nearest cluster centroid. Expected to reduce noise to < 12%.

## 7. Recommendations

1. **Immediate**: Implement outlier reduction to cut noise from 25.7% → < 12%
2. **Short-term**: Increase min_cluster_size to 20 to consolidate fragmented topics
3. **Medium-term**: Switch to Online BERTopic for BigQuery streaming integration
4. **Long-term**: Fine-tune LaBSE on Hinglish news data for better embeddings

## 8. Conclusion

The system successfully identifies major news themes from Hinglish titles. The primary bottleneck is the noise ratio, which is addressable through outlier reduction. The LLM labeling produces high-quality, search-ready topic names.
