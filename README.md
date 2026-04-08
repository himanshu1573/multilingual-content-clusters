# Social Media Content Clustering

This repo now focuses on one workflow: BERTopic-based clustering for multilingual social media titles, with optional local LLM labeling through Ollama.

## Main Files
- `social_listening_bertopic.py`: primary clustering pipeline
- `relabel_bertopic_outputs.py`: applies local LLM labels to an existing BERTopic run
- `BERTopic_WORKFLOW.md`: full workflow notes and results

## Installation
Use Python `3.10+` and install the dependencies:
```bash
pip install -r requirements.txt
```

For local LLM labels on an 8 GB Mac:
```bash
brew install ollama
brew services start ollama
ollama pull qwen2.5:3b
```

## Main Run
Run BERTopic on the Excel sheet:
```bash
.venv/bin/python social_listening_bertopic.py \
  --input "Video Titles.xlsx" \
  --input-type xlsx \
  --column title \
  --embedding-model sentence-transformers/LaBSE \
  --label-mode llm \
  --llm-provider ollama \
  --llm-model qwen2.5:3b \
  --min-cluster-size 20 \
  --min-samples 5 \
  --nr-topics auto \
  --output-dir outputs/bertopic_titles_labse_llm_clean \
  --no-visualize
```

Outputs:
- `outputs/bertopic_titles_labse_llm_clean/bertopic_clustered_documents.csv`
- `outputs/bertopic_titles_labse_llm_clean/bertopic_topic_summary.csv`
- `outputs/bertopic_titles_labse_llm_clean/bertopic_topic_summary.json`

## Relabel Only
To relabel an existing BERTopic output without rerunning embeddings:
```bash
.venv/bin/python relabel_bertopic_outputs.py \
  --output-dir outputs/bertopic_titles_labse_llm_clean \
  --llm-model qwen2.5:3b
```

## Current Best Run
- cleaned rows used: `8932`
- topics excluding noise: `27`
- noise rows: `2212`
- silhouette score: `0.0769`
- labels generated with local `qwen2.5:3b`

Detailed workflow notes live in `BERTopic_WORKFLOW.md`.
