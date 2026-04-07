"""
Social Listening - Content Clustering Script
=============================================
Groups social media posts by topic and auto-names each cluster.

Requirements:
    pip install pandas sentence-transformers hdbscan umap-learn scikit-learn plotly

Input:  A CSV file with at least one text column (e.g., 'post', 'text', 'content')
Output: - CSV with cluster labels added
        - Interactive HTML scatter plot (UMAP visualization)
        - Printed cluster names + sample posts
"""

import pandas as pd
import numpy as np
import os
import sys
from collections import Counter

# ── Config ────────────────────────────────────────────────────────────────────

INPUT_CSV      = "posts.csv"          # Path to your CSV file
TEXT_COLUMN    = "text"               # Column that contains the post text
OUTPUT_CSV     = "posts_clustered.csv"
OUTPUT_PLOT    = "clusters_plot.html"

MIN_CLUSTER_SIZE = 5                  # Minimum posts to form a cluster (tune this)
TOP_KEYWORDS     = 5                  # Keywords per cluster used for naming

# ── Step 1: Load data ─────────────────────────────────────────────────────────

def load_data(path: str, text_col: str) -> pd.DataFrame:
    print(f"📂 Loading data from: {path}")
    df = pd.read_csv(path)

    # Auto-detect text column if not found
    if text_col not in df.columns:
        candidates = [c for c in df.columns if df[c].dtype == object]
        if not candidates:
            sys.exit("❌ No text column found. Check TEXT_COLUMN setting.")
        text_col = candidates[0]
        print(f"⚠️  Column '{TEXT_COLUMN}' not found. Using '{text_col}' instead.")

    df = df[[text_col]].rename(columns={text_col: "text"}).dropna()
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 10].reset_index(drop=True)
    print(f"✅ Loaded {len(df)} posts.\n")
    return df


# ── Step 2: Embed posts ───────────────────────────────────────────────────────

def embed_posts(texts: list) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    print("🔢 Generating embeddings (this may take a minute)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")   # Fast, good quality
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)
    print(f"✅ Embeddings shape: {embeddings.shape}\n")
    return embeddings


# ── Step 3: Reduce dimensions (UMAP) ─────────────────────────────────────────

def reduce_dimensions(embeddings: np.ndarray, n_components=2) -> np.ndarray:
    import umap
    print("📉 Reducing dimensions with UMAP...")
    reducer = umap.UMAP(n_components=n_components, random_state=42, metric="cosine")
    reduced = reducer.fit_transform(embeddings)
    print("✅ Dimensionality reduction done.\n")
    return reduced


# ── Step 4: Cluster (HDBSCAN) ─────────────────────────────────────────────────

def cluster_posts(embeddings: np.ndarray, min_cluster_size: int) -> np.ndarray:
    import hdbscan
    print(f"🔍 Clustering with HDBSCAN (min_cluster_size={min_cluster_size})...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric="euclidean",
        cluster_selection_method="eom"
    )
    labels = clusterer.fit_predict(embeddings)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise    = (labels == -1).sum()
    print(f"✅ Found {n_clusters} clusters | {n_noise} posts unclustered (label=-1)\n")
    return labels


# ── Step 5: Name clusters via TF-IDF keywords ────────────────────────────────

def name_clusters(df: pd.DataFrame, top_n: int = TOP_KEYWORDS) -> dict:
    from sklearn.feature_extraction.text import TfidfVectorizer

    cluster_names = {}
    unique_labels = sorted(set(df["cluster"]))

    for label in unique_labels:
        if label == -1:
            cluster_names[-1] = "🔘 Unclustered / Noise"
            continue

        posts = df[df["cluster"] == label]["text"].tolist()
        if len(posts) < 2:
            cluster_names[label] = f"Cluster {label}"
            continue

        try:
            tfidf = TfidfVectorizer(stop_words="english", max_features=200, ngram_range=(1, 2))
            tfidf.fit_transform(posts)
            scores = zip(tfidf.get_feature_names_out(), tfidf.idf_)
            # Lower IDF = more common in this cluster
            sorted_terms = sorted(scores, key=lambda x: x[1])
            keywords = [t for t, _ in sorted_terms[:top_n]]
            cluster_names[label] = " · ".join(keywords).title()
        except Exception:
            cluster_names[label] = f"Cluster {label}"

    return cluster_names


# ── Step 6: Visualize ─────────────────────────────────────────────────────────

def visualize(df: pd.DataFrame, output_path: str):
    import plotly.express as px
    print("🎨 Generating interactive visualization...")

    fig = px.scatter(
        df,
        x="umap_x",
        y="umap_y",
        color="cluster_name",
        hover_data={"text": True, "cluster": True, "umap_x": False, "umap_y": False},
        title="Social Listening — Content Clusters",
        labels={"cluster_name": "Cluster"},
        template="plotly_white",
        width=1100,
        height=700,
    )
    fig.update_traces(marker=dict(size=6, opacity=0.7))
    fig.write_html(output_path)
    print(f"✅ Plot saved to: {output_path}\n")


# ── Step 7: Print summary ──────────────────────────────────────────────────────

def print_summary(df: pd.DataFrame):
    print("=" * 60)
    print("  CLUSTER SUMMARY")
    print("=" * 60)
    for name, group in df.groupby("cluster_name"):
        count = len(group)
        sample = group["text"].iloc[0][:120].replace("\n", " ")
        print(f"\n📌 {name}  ({count} posts)")
        print(f"   Sample: \"{sample}...\"")
    print("\n" + "=" * 60)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    # Check input file
    if not os.path.exists(INPUT_CSV):
        print(f"⚠️  '{INPUT_CSV}' not found. Creating a sample CSV for demo...\n")
        sample_posts = [
            "The new iPhone 16 camera is absolutely stunning, best photos I've taken",
            "Apple's latest phone has incredible photo quality",
            "Shot this sunset on iPhone 16 Pro — the colors are unreal",
            "The battery life on this phone is terrible, dies by noon",
            "My phone battery drains so fast after the latest update",
            "Why does the battery die so quickly on iOS 18?",
            "Customer support was helpful and resolved my issue quickly",
            "Great experience with the support team, very responsive",
            "The helpdesk got back to me within hours, impressed",
            "Price hike is ridiculous, not worth the upgrade",
            "Too expensive compared to Android options",
            "Overpriced for what you get, Samsung is better value",
            "Love the new design, sleek and premium feel",
            "The titanium finish looks and feels amazing",
            "Best looking phone I've ever owned",
        ] * 4  # Repeat to hit min_cluster_size
        pd.DataFrame({"text": sample_posts}).to_csv(INPUT_CSV, index=False)
        print(f"✅ Sample CSV created: {INPUT_CSV}\n")

    df = load_data(INPUT_CSV, TEXT_COLUMN)
    embeddings = embed_posts(df["text"].tolist())
    reduced    = reduce_dimensions(embeddings)
    labels     = cluster_posts(reduced, MIN_CLUSTER_SIZE)

    df["cluster"] = labels
    df["umap_x"]  = reduced[:, 0]
    df["umap_y"]  = reduced[:, 1]

    cluster_names        = name_clusters(df)
    df["cluster_name"]   = df["cluster"].map(cluster_names)

    print_summary(df)

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"💾 Clustered data saved to: {OUTPUT_CSV}")

    visualize(df, OUTPUT_PLOT)
    print("🚀 Done! Open the HTML file in your browser to explore clusters interactively.")


if __name__ == "__main__":
    main()
