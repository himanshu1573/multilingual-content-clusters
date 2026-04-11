# BERTopic Workflow

This document records the BERTopic-based workflow that is now available in this repo and the exact setup used for the Excel-sheet clustering run.

## Purpose

The goal is to cluster social-listening text into semantic topics and produce outputs that are easier to inspect than raw titles alone.

This pipeline is intended for:
- multilingual short text
- noisy social media titles
- English, Hindi, and Hinglish content
- topic discovery without forcing a fixed number of clusters

## Files Used

- [`social_listening_bertopic.py`](/Users/himanshup/Clustering%20/social_listening_bertopic.py): main BERTopic pipeline
- [`Video Titles.xlsx`](/Users/himanshup/Clustering%20/Video%20Titles.xlsx): current sheet-based input dataset
- [`requirements.txt`](/Users/himanshup/Clustering%20/requirements.txt): Python dependencies
- [`README.md`](/Users/himanshup/Clustering%20/README.md): repo-level usage notes

## Components Used

### 1. Input Loading

The BERTopic script supports:
- JSON exports with `posts[].content`
- CSV files
- XLSX files

For the current sheet workflow:
- input file: `Video Titles.xlsx`
- input type: `xlsx`
- text column: `title`

### 2. Text Cleaning

The script normalizes noisy text before embedding:
- lowercases text
- removes URLs
- removes mentions
- converts hashtags into plain tokens
- converts punctuation separators into spaces instead of merging words
- collapses extra spaces
- removes common stopwords and recurring boilerplate tokens
- deduplicates on cleaned text instead of raw formatting only

This is important because raw social data often contains repeated platform words that would otherwise distort topics.

Important fix:
- the cleaning now preserves multilingual token boundaries better, so mixed titles no longer collapse into merged tokens such as `warईरान` or `usisrael`

### 3. Embeddings

Embedding model used for the sheet run:
- `sentence-transformers/LaBSE`

Why this model:
- works well for multilingual text
- handles Hindi, English, and mixed Hinglish better than English-only embedding models

### 4. Topic Modeling Stack

The BERTopic pipeline uses:
- `BERTopic` as the overall topic-modeling framework
- `UMAP` for dimensionality reduction
- `HDBSCAN` for clustering
- `CountVectorizer` for c-TF-IDF vocabulary extraction
- `KeyBERTInspired` and `MaximalMarginalRelevance` for improved topic representation

### 5. Labeling Modes

Two label modes are supported:

- `keywords`
  - default mode
  - uses BERTopic-derived keywords
  - no local LLM required

- `llm`
  - optional mode
  - supports local LLM labeling through `ollama` or `transformers`
  - includes cache support
  - currently falls back safely to keyword labels if the LLM is unavailable

## Best LLM Choice For An 8 GB Mac

Given your machine constraints:
- RAM: `8 GB`
- storage: `256 GB`

The best practical choice is:
- `ollama` as the local inference backend
- `qwen2.5:3b` as the first model to try

Why this is the best fit:
- smaller and more realistic than a 13B model on 8 GB RAM
- better multilingual support than many tiny English-first models
- better suited to Hindi, English, and Hinglish title labeling
- much more practical on macOS than loading a large Hugging Face causal model directly in Python

Recommended fallback options:
- `llama3.2:3b`
  - good if you specifically want a Meta-family local model
  - also supports Hindi officially in Ollama’s model listing
- `phi3:mini`
  - lightweight and fast
  - less suitable for your multilingual title mix because it is primarily intended for English use cases

Recommendation order for your sheet:
1. `qwen2.5:3b`
2. `llama3.2:3b`
3. `phi3:mini`

Avoid on this machine:
- `Llama 2 13B`
- `7B+` models through raw Transformers/PyTorch

Those are too heavy for comfortable local use on an 8 GB Mac and will usually be slow or memory-constrained.

## Exact Sheet Run

Command used:

```bash
.venv/bin/python social_listening_bertopic.py \
  --input "Video Titles.xlsx" \
  --input-type xlsx \
  --column title \
  --embedding-model sentence-transformers/LaBSE \
  --label-mode keywords \
  --min-cluster-size 15 \
  --nr-topics auto \
  --output-dir outputs/bertopic_titles_labse \
  --no-visualize
```

## Exact Settings Used

- input file: `Video Titles.xlsx`
- input type: `xlsx`
- text column: `title`
- embedding model: `sentence-transformers/LaBSE`
- label mode: `keywords`
- minimum cluster size: `15`
- topic reduction: `auto`
- visualization: skipped in this run

## Result Summary For The Sheet

From [`outputs/bertopic_titles_labse/bertopic_topic_summary.json`](/Users/himanshup/Clustering%20/outputs/bertopic_titles_labse/bertopic_topic_summary.json):

- cleaned documents used: `9009`
- topic count excluding noise: `50`
- total topic rows including noise: `51`
- noise documents: `1972`
- noise ratio: `0.2189`
- silhouette score: `0.0478`
- calinski-harabasz score: `45.95`

BERTopic also reported:
- reduced raw topics from `158` to `51`

## Cleaned + Stricter Run

After improving multilingual cleaning and rerunning with stricter clustering:

Command used:

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

Run summary from [`outputs/bertopic_titles_labse_llm_clean/bertopic_topic_summary.json`](/Users/himanshup/Clustering%20/outputs/bertopic_titles_labse_llm_clean/bertopic_topic_summary.json):

- cleaned documents used: `8932`
- topic count excluding noise: `27`
- noise documents: `2212`
- noise ratio: `0.2476`
- silhouette score: `0.0769`
- calinski-harabasz score: `63.61`
- reduced raw topics from `93` to `28`

What improved:
- topic count is much tighter
- silhouette score improved from `0.0478` to `0.0769`
- calinski-harabasz score improved from `45.95` to `63.61`
- LLM topic names are substantially more readable

Example LLM labels from the cleaned run:
- `Iran-Israel War`
- `Elections 2026`
- `Bulldozers Violence Maharashtra`
- `Security Breach Delhi Assembly`
- `LPG Crisis`
- `Political Leaders Congress BJP`

## Top Topics From The Sheet Run

- Topic `0`: `Suhail Iran Israel / Syed Suhail Iran / Israel Iran War` with `3408` documents
- Topic `1`: `Bengal Elections 2026 / Bengal Election 2026 / Election 2026 Bjp` with `928` documents
- Topic `2`: `Viralvideo Viralnews Viral / Viralnews Viralvideo Viral / Iitbaba Viralvideo` with `619` documents
- Topic `3`: `Choudhary / Chaudhary / Iranisraelwar Trump` with `325` documents
- Topic `4`: `Delhi Assembly Security / Security Lapse Delhi / Delhi Assembly नसभ` with `190` documents

## Outputs Produced

This run produced:

- [`outputs/bertopic_titles_labse/bertopic_clustered_documents.csv`](/Users/himanshup/Clustering%20/outputs/bertopic_titles_labse/bertopic_clustered_documents.csv)
  - each title with its assigned topic id and topic label

- [`outputs/bertopic_titles_labse/bertopic_topic_summary.csv`](/Users/himanshup/Clustering%20/outputs/bertopic_titles_labse/bertopic_topic_summary.csv)
  - topic-level summary table

- [`outputs/bertopic_titles_labse/bertopic_topic_summary.json`](/Users/himanshup/Clustering%20/outputs/bertopic_titles_labse/bertopic_topic_summary.json)
  - machine-readable run summary plus topics

## Notes On Interpretation

- BERTopic does not force every title into a clean topic, so some rows remain as `Outliers / Noise`.
- Large topics can still reflect dominant news cycles and repeated headline styles.
- Keyword labels are useful, but optional LLM labels can later improve readability for presentation.

## Recommended Next Runs

If we continue focusing on the Excel sheet, the next useful comparisons are:

```bash
.venv/bin/python social_listening_bertopic.py \
  --input "Video Titles.xlsx" \
  --input-type xlsx \
  --column title \
  --embedding-model sentence-transformers/LaBSE \
  --label-mode keywords \
  --min-cluster-size 20 \
  --nr-topics auto \
  --output-dir outputs/bertopic_titles_labse_mcs20
```

```bash
.venv/bin/python social_listening_bertopic.py \
  --input "Video Titles.xlsx" \
  --input-type xlsx \
  --column title \
  --embedding-model sentence-transformers/LaBSE \
  --label-mode keywords \
  --min-cluster-size 15 \
  --output-dir outputs/bertopic_titles_labse_viz
```

The first command helps reduce micro-topics. The second adds the visualization HTML.

## How To Use LLM Labeling On This Mac

Recommended local setup:

1. Install Ollama on the Mac.
2. Pull a small model:

```bash
ollama pull qwen2.5:3b
```

3. Run BERTopic with Ollama-backed labeling:

```bash
.venv/bin/python social_listening_bertopic.py \
  --input "Video Titles.xlsx" \
  --input-type xlsx \
  --column title \
  --embedding-model sentence-transformers/LaBSE \
  --label-mode llm \
  --llm-provider ollama \
  --llm-model qwen2.5:3b \
  --min-cluster-size 15 \
  --nr-topics auto \
  --output-dir outputs/bertopic_titles_labse_llm
```

If Ollama is not running or the model is missing, the script now falls back to keyword labels instead of failing.

## Relabel Existing Outputs

To avoid rerunning the full embedding pipeline when only labels need to change, use:

```bash
.venv/bin/python relabel_bertopic_outputs.py \
  --output-dir outputs/bertopic_titles_labse_llm_clean \
  --llm-model qwen2.5:3b
```

This reads the existing topic summary and clustered CSV, applies Ollama labels, and updates the output files in place.

## How To Reduce Noise Further

The current sheet run still has noise because:
- many titles are short and repetitive
- some titles are near-duplicates
- some topics are driven by recurring channel/person names
- some viral/hashtag-heavy titles have low semantic value

Best improvements to try next:

### 1. Increase Cluster Strictness

Try:
- `--min-cluster-size 20`
- `--min-cluster-size 25`

Why:
- reduces tiny or weak clusters
- pushes more borderline titles into noise instead of low-quality topics

### 2. Add More Domain Stopwords

Useful examples for this dataset:
- repeated channel names
- generic words like `viralvideo`, `viralnews`, `breakingnews`
- repeated anchor names if they dominate many titles

Why:
- improves topic names
- reduces clusters based on branding instead of subject matter

### 3. Remove Near-Duplicates

Current script removes exact duplicate raw texts only. A better next step is near-duplicate removal for titles that differ only by:
- hashtags
- punctuation
- extra emoji
- minor formatting changes

Why:
- large duplicate-heavy events can dominate topic discovery

### 4. Keep `LaBSE` For The Sheet

Do not switch the sheet workflow to an English-only embedder unless the dataset changes.

Why:
- this sheet contains Hindi and Hinglish
- `LaBSE` is better aligned with the data than `all-MiniLM-L6-v2`

### 5. Use LLM Labels Only After Topic Quality Is Reasonable

The LLM improves readability, not clustering quality itself.

Meaning:
- BERTopic + embeddings decide which titles group together
- the LLM only renames those groups

So if the clusters are noisy, the first fixes should be preprocessing and clustering parameters.
