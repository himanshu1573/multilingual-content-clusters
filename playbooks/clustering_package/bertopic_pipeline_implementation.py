#!/usr/bin/env python3
"""
Hinglish News Content Clustering Pipeline
Production-ready BERTopic clustering for mixed Hindi+English video titles.
Usage: python bertopic_pipeline_implementation.py "Video Titles.txt" outputs/
"""
import argparse
import html
import json
import re
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer
from sklearn.metrics import silhouette_score

# ─── Stopwords ───────────────────────────────────────────────────────────────
DOMAIN_STOPS = {
    "abp","amp","breaking","breakingnews","channel","click","com","exclusive",
    "facebook","follow","hindi","http","https","instagram","latest","live",
    "mobile","ndtv","news","playlist","share","shorts","shots","shows",
    "subscribe","today","tonight","tweet","updates","video","videos","watch",
    "whatsapp","www","xcom","viral","viralnews","viralvideo","ytshorts","youtube",
    "aajtak","aajtakdigital","ndtvindia","timesnownavbharat","tv9","tv9d",
    "news18","n18s","topnews","shortvideo","shortsvideo","viralshorts","political",
    "at2","atshorts","n18v","upnews",
}
HINDI_STOPS = {
    "में","के","की","का","पर","ने","को","से","है","क्या",
    "हो","था","थी","थे","गया","गई","गए","कर","किया","दिया","लिया",
    "होना","करना","लेकिन","मगर","और","भी","ही","कि","जो","तो",
    "ये","वे","वह","इस","उस","उसे","इसे","इसकी","उसकी","उनका",
    "अपना","अपने","अपनी","सब","कोई","कुछ","बहुत","कम","ज्यादा",
    "वही","सही","गलत","सा","सी","तक","लिए","बारे","बीच","वाले",
    "वाली","वाला","हैं","रहे","रही","रहा","सकते","सकता","सकती",
}
MONTHS = {"january","february","march","april","may","june","july","august",
           "september","october","november","december"}
STOPS = set(ENGLISH_STOP_WORDS) | MONTHS | DOMAIN_STOPS | HINDI_STOPS

# ─── Entity Normalization ────────────────────────────────────────────────────
CANONICAL_PATTERNS = [
    (re.compile(r"\brahul\s+gandhi\b|\brahulgandhi\b", re.I), " rahul_gandhi "),
    (re.compile(r"\bpriyanka\s+gandhi\b|\bpriyankagandhi\b", re.I), " priyanka_gandhi "),
    (re.compile(r"\bakhilesh\s+yadav\b|\bakhileshyadav\b", re.I), " akhilesh_yadav "),
    (re.compile(r"\bdonald\s+trump\b", re.I), " donald_trump "),
    (re.compile(r"\bpm\s+modi\b|\bmodi\s+ji\b|\bnarendra\s+modi\b", re.I), " pm_modi "),
    (re.compile(r"\bhimanta\s+biswa\s+sarma\b|\bhimantabiswasarma\b", re.I), " himanta_biswa_sarma "),
    (re.compile(r"\bpawan\s+khera\b", re.I), " pawan_khera "),
    (re.compile(r"\bsandeep\s+chaudhary\b", re.I), " sandeep_chaudhary "),
    (re.compile(r"\bsushant\s+sinha\b", re.I), " sushant_sinha "),
    (re.compile(r"\biit\s+baba\b|\biitbaba\b", re.I), " iit_baba "),
    (re.compile(r"\bbjp\b", re.I), " bjp "),
    (re.compile(r"\bcongress\b", re.I), " congress "),
    (re.compile(r"\baap\b", re.I), " aap "),
]

# ─── Regex patterns ─────────────────────────────────────────────────────────
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#(\w+)")
SEPARATOR_RE = re.compile(r"[-/|:+]+")
CAMEL_RE = re.compile(r"(?<=[a-z])(?=[A-Z])")
NON_WORD_RE = re.compile(r"[^\w\s\u0900-\u097F]")
MULTISPACE_RE = re.compile(r"\s+")
AC_CODE_RE = re.compile(r"AC/\d{1,4}/\d{1,4}", re.I)


def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    text = AC_CODE_RE.sub(" ", text)
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = HASHTAG_RE.sub(r" \1 ", text)
    text = CAMEL_RE.sub(" ", text)
    text = SEPARATOR_RE.sub(" ", text)
    text = text.lower().strip()
    text = NON_WORD_RE.sub(" ", text)
    tokens = []
    for tok in text.split():
        if tok.isascii() and len(tok) <= 2:
            continue
        if tok in STOPS:
            continue
        tokens.append(tok)
    return MULTISPACE_RE.sub(" ", " ".join(tokens)).strip()


def normalize_entities(text: str) -> str:
    value = html.unescape(text or "")
    for pat, repl in CANONICAL_PATTERNS:
        value = pat.sub(repl, value)
    return value


def load_data(path: Path, column: str = "title") -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    elif suffix == ".json":
        df = pd.DataFrame(json.loads(path.read_text()))
    else:
        try:
            df = pd.read_csv(path)
        except Exception:
            df = pd.read_csv(path, sep="\v", engine="python")

    if column in df.columns:
        text_col = column
    else:
        text_col = [c for c in df.columns if df[c].dtype == object][0]

    out = pd.DataFrame()
    out["raw_text"] = df[text_col].fillna("").astype(str).str.strip()
    out = out[out["raw_text"] != ""].reset_index(drop=True)
    out["clean_text"] = out["raw_text"].map(normalize_text)
    out["ner_text"] = out["raw_text"].map(lambda t: normalize_text(normalize_entities(t)))
    out = out[out["clean_text"] != ""].reset_index(drop=True)
    return out


def build_and_run(docs_df: pd.DataFrame, min_cluster_size: int = 15,
                  min_samples: int = 5, nr_topics: str = "auto"):
    from bertopic import BERTopic
    from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance
    import hdbscan
    import umap

    print(f"Loaded {len(docs_df)} cleaned documents")

    # Embed
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer("sentence-transformers/LaBSE")
    docs = docs_df["ner_text"].tolist()
    embeddings = embedder.encode(docs, show_progress_bar=True, batch_size=32)
    print("Embeddings complete")

    # Build model
    umap_model = umap.UMAP(n_neighbors=15, n_components=5, min_dist=0.0,
                           metric="cosine", random_state=42)
    hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                                     min_samples=min_samples, metric="euclidean",
                                     cluster_selection_method="eom", prediction_data=True)
    vectorizer = CountVectorizer(stop_words="english", ngram_range=(1, 3), min_df=2, max_df=0.95)
    representation = {"Main": KeyBERTInspired(), "MMR": MaximalMarginalRelevance(diversity=0.3)}

    topic_model = BERTopic(
        embedding_model=embedder, umap_model=umap_model, hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer, representation_model=representation,
        verbose=True, nr_topics=nr_topics, top_n_words=10,
    )

    topics, _ = topic_model.fit_transform(docs, embeddings)
    docs_df["topic_id"] = topics
    print(f"Found {len(set(topics)) - (1 if -1 in topics else 0)} topics")

    return docs_df, topic_model, embeddings, topics


def label_topics_with_ollama(topic_model, model_name="qwen2.5:3b",
                              host="http://localhost:11434"):
    label_map = {}
    for topic_id in sorted(set(topic_model.get_topics().keys())):
        if topic_id == -1:
            label_map[-1] = "Outliers / Noise"
            continue
        kw_pairs = topic_model.get_topic(topic_id) or []
        keywords = [w for w, _ in kw_pairs[:8]]
        rep_docs = (topic_model.get_representative_docs(topic_id) or [])[:3]
        fallback = " / ".join(w.replace("_", " ") for w in keywords[:3]).title()

        try:
            doc_block = "\n".join(f"{i}. {d[:280]}" for i, d in enumerate(rep_docs, 1))
            prompt = (f"You label social listening topics.\n"
                      f"Return only a short factual label with 2 to 6 words.\n\n"
                      f"Keywords: {', '.join(keywords)}\n"
                      f"Representative posts:\n{doc_block}\n\nLabel:")
            resp = requests.post(f"{host}/api/generate",
                                 json={"model": model_name, "prompt": prompt,
                                        "stream": False, "options": {"temperature": 0.3}},
                                 timeout=120)
            resp.raise_for_status()
            label = resp.json().get("response", fallback).strip().strip('"').strip("'")
            if "label:" in label.lower():
                label = label.split(":", 1)[-1].strip()
            label = label.splitlines()[0].strip()[:80] or fallback
        except Exception:
            label = fallback

        label_map[topic_id] = label
        print(f"  Topic {topic_id}: {label}")
    return label_map


def save_outputs(output_dir: Path, docs_df: pd.DataFrame, topic_model, label_map: dict):
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    docs_df["topic_label"] = docs_df["topic_id"].map(label_map)
    cols = ["raw_text", "clean_text", "ner_text", "topic_id", "topic_label"]
    docs_df[cols].to_csv(output_dir / "clustered_documents.csv", index=False)

    # Summary JSON
    summary = {"topics": []}
    for tid in sorted(label_map.keys()):
        if tid == -1:
            continue
        count = int((docs_df["topic_id"] == tid).sum())
        kws = [w for w, _ in (topic_model.get_topic(tid) or [])[:10]]
        summary["topics"].append({"topic_id": tid, "label": label_map[tid],
                                   "document_count": count, "keywords": kws})
    summary["noise_count"] = int((docs_df["topic_id"] == -1).sum())
    summary["total_documents"] = len(docs_df)
    (output_dir / "cluster_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))

    print(f"\nOutputs saved to {output_dir}/")
    print(f"  clustered_documents.csv ({len(docs_df)} rows, {len(cols)} columns)")
    print(f"  cluster_summary.json ({len(summary['topics'])} topics)")


def main():
    parser = argparse.ArgumentParser(description="Hinglish BERTopic Clustering Pipeline")
    parser.add_argument("input", help="Path to input file (txt/csv/xlsx)")
    parser.add_argument("output_dir", nargs="?", default="outputs/clusters",
                        help="Output directory")
    parser.add_argument("--column", default="title", help="Text column name")
    parser.add_argument("--min-cluster-size", type=int, default=15)
    parser.add_argument("--llm-model", default="qwen2.5:3b")
    args = parser.parse_args()

    start = time.time()
    docs_df = load_data(Path(args.input), args.column)
    docs_df, topic_model, embeddings, topics = build_and_run(
        docs_df, min_cluster_size=args.min_cluster_size)

    print("\nLabeling topics with LLM...")
    label_map = label_topics_with_ollama(topic_model, model_name=args.llm_model)

    save_outputs(Path(args.output_dir), docs_df, topic_model, label_map)
    elapsed = time.time() - start
    noise_pct = round(100 * (docs_df["topic_id"] == -1).mean(), 1)
    print(f"\nDone in {elapsed:.0f}s. Noise: {noise_pct}%")


if __name__ == "__main__":
    main()
