# Project Setup Guide

## Overview

This project clusters multilingual (Hindi+English) video titles into semantic topics using BERTopic. It is designed for social listening and news content categorization.

## Prerequisites

- **Python**: 3.10+
- **Hardware**: Apple Silicon Mac (8GB+) or any machine with 8GB+ RAM
- **Ollama**: Required for LLM-based topic labeling

## Installation

### 1. Python Environment

```bash
cd /path/to/project
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If `requirements.txt` is missing:
```bash
pip install sentence-transformers bertopic hdbscan umap-learn pandas openpyxl requests scikit-learn
```

### 2. Ollama (Local LLM)

```bash
# macOS
brew install ollama
brew services start ollama
ollama pull qwen2.5:3b

# Verify
ollama list   # Should show qwen2.5:3b
curl http://localhost:11434/api/tags   # Should return JSON
```

### 3. Embedding Model (First Run)

The first run downloads the LaBSE model (~1.8GB). After that, it is cached locally.

```bash
# Pre-download (optional)
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/LaBSE')"
```

## Usage

### Quick Run
```bash
python clustering_package/bertopic_pipeline_implementation.py "Video Titles.txt" outputs/granular_experiment
```

### Full Options
```bash
python clustering_package/bertopic_pipeline_implementation.py \
  "Video Titles.txt" \
  outputs/granular_experiment \
  --column title \
  --min-cluster-size 15 \
  --llm-model qwen2.5:3b
```

### Using the Existing Pipeline
```bash
python social_listening_bertopic.py \
  --input "Video Titles.txt" \
  --input-type csv \
  --column title \
  --embedding-model sentence-transformers/LaBSE \
  --label-mode llm \
  --llm-provider ollama \
  --llm-model qwen2.5:3b \
  --min-cluster-size 15 \
  --min-samples 5 \
  --nr-topics auto \
  --output-dir outputs/granular_experiment \
  --no-visualize
```

## Output Files

After running, the output directory contains:

| File | Content |
|------|---------|
| `clustered_documents.csv` | 5 columns: raw_text, clean_text, ner_text, topic_id, topic_label |
| `cluster_summary.json` | Topic list with keywords, document counts |
| `bertopic_topic_summary.csv` | Detailed topic breakdown |
| `llm_label_cache.json` | Cached LLM labels (avoids re-querying) |

## Project File Structure

```
Clustering/
├── clustering_package/          # This package
│   ├── 00_START_HERE.txt
│   ├── bertopic_configuration.json
│   ├── bertopic_pipeline_implementation.py
│   ├── ML_Optimization_Report.md
│   ├── Implementation_Guide.md
│   ├── QUICK_RECOMMENDATIONS.txt
│   ├── EXACT_CODE_CHANGES.txt
│   └── README_PROJECT_SETUP.md
├── social_listening_bertopic.py  # Main pipeline (1100+ lines)
├── ner_extraction.py             # NER extraction utility
├── expert_clustering_engine.py   # Bucketed clustering variant
├── compare_topic_strategies.py   # Strategy comparison tool
├── Video Titles.txt              # Input data (9,722 titles)
├── Video Titles.xlsx             # Input data (Excel format)
├── requirements.txt              # Python dependencies
├── outputs/                      # Clustering results
│   └── granular_experiment/
│       ├── bertopic_clustered_documents.csv
│       ├── bertopic_topic_summary.json
│       └── llm_label_cache.json
└── .venv/                        # Python virtual environment
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: bertopic` | Run `pip install bertopic` |
| Ollama connection refused | Run `brew services start ollama` |
| Model not found | Run `ollama pull qwen2.5:3b` |
| CSV parse error | Script auto-falls back to `\v` separator |
| High noise (> 15%) | Decrease `min_cluster_size` to 10 |
| Slow embeddings | Normal: ~90s for 9k docs on Apple Silicon |

## Future: BigQuery Integration

When moving from local files to BigQuery streaming:

1. Replace `pd.read_csv()` with `bigquery.Client().query().to_dataframe()`
2. Use Online BERTopic for incremental learning on new batches
3. Implement cosine similarity assignment instead of HDBSCAN for new docs
4. Labels generated "on the go" as each batch arrives
