# Multilingual E5 BERTopic

This folder is a fresh clustering track built from the BERTopic pipeline without modifying older approaches in the repository.

## Goal

Use `intfloat/multilingual-e5-large` as the primary multilingual embedder for clustering noisy Hindi, English, and Hinglish news titles with BERTopic and explicit outlier reduction.

## Why This Exists

The previous `frequency + NER` flow was useful for recurring entity combinations, but it is brittle for:

- paraphrased titles about the same story
- titles with weak or missing named entities
- event-heavy headlines where the action matters more than the entities

This approach is semantic-first:

1. clean and normalize noisy titles
2. embed with `multilingual-e5-large`
3. cluster with `BERTopic + UMAP + HDBSCAN`
4. reduce outliers with embedding-based reassignment
5. optionally relabel topics with a local LLM

## What Changed In This Fresh Track

- created a new standalone script: [multilingual_e5_bertopic.py](/Users/himanshup/Clustering%20/07_Multilingual_E5_BERTopic/multilingual_e5_bertopic.py)
- default embedding model is now `intfloat/multilingual-e5-large`
- added E5-specific `passage:` prefixing before embedding
- main flow now uses the existing helper that performs BERTopic outlier reduction
- run summaries now record the embedding prefix and outlier reduction strategy

## Recommended First Run

```bash
python 07_Multilingual_E5_BERTopic/multilingual_e5_bertopic.py \
  --input "00_Shared_Data/Video Titles.xlsx" \
  --input-type xlsx \
  --column title \
  --label-mode keywords \
  --min-cluster-size 15 \
  --min-samples 3 \
  --nr-topics auto \
  --output-dir outputs/multilingual_e5_bertopic_run \
  --no-visualize
```

## Optional LLM Labeling Run

```bash
python 07_Multilingual_E5_BERTopic/multilingual_e5_bertopic.py \
  --input "00_Shared_Data/Video Titles.xlsx" \
  --input-type xlsx \
  --column title \
  --label-mode llm \
  --llm-provider ollama \
  --llm-model qwen2.5:3b \
  --min-cluster-size 15 \
  --min-samples 3 \
  --nr-topics auto \
  --output-dir outputs/multilingual_e5_bertopic_llm \
  --no-visualize
```

## Expected Outputs

- `bertopic_clustered_documents.csv`
- `bertopic_topic_summary.csv`
- `bertopic_topic_summary.json`
- optional `bertopic_visualization.html`

## Notes

- `multilingual-e5-large` may need a first-time model download if it is not already cached.
- For E5 models, document text is embedded with a `passage:` prefix.
- This folder is intended to be the clean starting point for the next round of clustering evaluation.
