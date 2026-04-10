# Implementation Guide: Hinglish BERTopic Clustering

## Step-by-Step Code Reference

### 1. Environment Setup

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install sentence-transformers bertopic hdbscan umap-learn pandas openpyxl requests

# Install Ollama for LLM labeling (macOS)
brew install ollama
brew services start ollama
ollama pull qwen2.5:3b
```

### 2. Data Loading

```python
import pandas as pd
from pathlib import Path

def load_data(path: Path, column: str = "title") -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        try:
            df = pd.read_csv(path)
        except Exception:
            # Fallback: treat entire line as single column (handles commas in titles)
            df = pd.read_csv(path, sep="\v", engine="python")
    
    out = pd.DataFrame()
    out["raw_text"] = df[column].fillna("").astype(str).str.strip()
    out = out[out["raw_text"] != ""].reset_index(drop=True)
    return out
```

### 3. Text Cleaning for Hinglish

```python
import re, html

# Key insight: Devanagari characters live in Unicode U+0900-U+097F
# Our regex PRESERVES them while stripping Latin noise
NON_WORD_RE = re.compile(r"[^\w\s\u0900-\u097F]")

def normalize_text(text: str) -> str:
    text = html.unescape(text)
    text = re.sub(r"https?://\S+", " ", text)       # URLs
    text = re.sub(r"@\w+", " ", text)                # Mentions
    text = re.sub(r"#(\w+)", r" \1 ", text)          # Hashtags -> words
    text = text.lower()
    text = NON_WORD_RE.sub(" ", text)                # Strip punctuation, keep Hindi

    tokens = []
    for tok in text.split():
        # CRITICAL: Only filter short ASCII tokens
        # Hindi tokens like "हो" (2 chars) are meaningful
        if tok.isascii() and len(tok) <= 2:
            continue
        if tok in STOPS:
            continue
        tokens.append(tok)
    return " ".join(tokens).strip()
```

### 4. Entity Normalization

```python
# Map entity variants to canonical forms
PATTERNS = [
    (re.compile(r"\bpm\s+modi\b|\bmodi\s+ji\b|\bnarendra\s+modi\b", re.I), " pm_modi "),
    (re.compile(r"\brahul\s+gandhi\b|\brahulgandhi\b", re.I), " rahul_gandhi "),
    (re.compile(r"\bdonald\s+trump\b", re.I), " donald_trump "),
    # Add more as needed...
]

def normalize_entities(text: str) -> str:
    for pat, repl in PATTERNS:
        text = pat.sub(repl, text)
    return text

# Apply: first normalize entities, then clean text
df["ner_text"] = df["raw_text"].map(lambda t: normalize_text(normalize_entities(t)))
```

### 5. Embedding with LaBSE

```python
from sentence_transformers import SentenceTransformer

# LaBSE understands 109 languages including Hindi
embedder = SentenceTransformer("sentence-transformers/LaBSE")

# It knows "Iran attack" ≈ "ईरान पर हमला"
docs = df["ner_text"].tolist()
embeddings = embedder.encode(docs, show_progress_bar=True, batch_size=32)
# Shape: (8930, 768)
```

### 6. BERTopic Clustering

```python
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
import hdbscan, umap

umap_model = umap.UMAP(n_neighbors=15, n_components=5, min_dist=0.0,
                        metric="cosine", random_state=42)

hdbscan_model = hdbscan.HDBSCAN(
    min_cluster_size=15,    # Minimum docs to form a cluster
    min_samples=5,          # Core point density
    metric="euclidean",
    cluster_selection_method="eom",
    prediction_data=True,
)

topic_model = BERTopic(
    embedding_model=embedder,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    vectorizer_model=CountVectorizer(stop_words="english", ngram_range=(1,3)),
    representation_model={"Main": KeyBERTInspired()},
    nr_topics="auto",
)

topics, _ = topic_model.fit_transform(docs, embeddings)
```

### 7. LLM Labeling with Ollama

```python
import requests

def label_topic(keywords, rep_docs, model="qwen2.5:3b"):
    prompt = (
        "You label social listening topics.\n"
        "Return only a short factual label with 2 to 6 words.\n\n"
        f"Keywords: {', '.join(keywords[:8])}\n"
        f"Representative posts:\n" +
        "\n".join(f"{i}. {d[:280]}" for i, d in enumerate(rep_docs[:3], 1)) +
        "\n\nLabel:"
    )
    resp = requests.post("http://localhost:11434/api/generate",
                         json={"model": model, "prompt": prompt, "stream": False},
                         timeout=120)
    return resp.json()["response"].strip()
```

### 8. Simplified Output

```python
# Only 5 columns in final CSV
cols = ["raw_text", "clean_text", "ner_text", "topic_id", "topic_label"]
df[cols].to_csv("clustered_documents.csv", index=False)
```

## Parameter Tuning Guide

| Parameter | Default | Effect of Increase | Effect of Decrease |
|-----------|---------|-------------------|-------------------|
| min_cluster_size | 15 | Fewer, larger clusters. More noise. | More, smaller clusters. Less noise. |
| min_samples | 5 | Stricter density. More noise. | Looser density. Less noise. |
| n_neighbors (UMAP) | 15 | Global structure preserved. | Local structure preserved. |
| nr_topics | "auto" | Set to int (e.g., 20) to force merge. | N/A |

## Future: BigQuery Integration

```python
from google.cloud import bigquery

client = bigquery.Client()
query = "SELECT title FROM `project.dataset.video_titles` LIMIT 10000"
df = client.query(query).to_dataframe()
# Then feed into the same pipeline above
```
