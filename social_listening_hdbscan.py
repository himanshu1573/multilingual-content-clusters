import os
import re
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
import umap
import hdbscan
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px

# Configuration
SIMILARITY_THRESHOLD = 0.60
MIN_CLUSTER_SIZE = 5
MODEL_NAME = "LaBSE"
STATE_DIR = Path("cluster_state")
OUTPUT_DIR = Path("outputs")

def normalize_hinglish(text):
    """Clean and normalize Hinglish/English/Hindi text."""
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    # Keep alphanumeric and Devanagari characters
    text = re.sub(r'[^\w\s\u0900-\u097F]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def get_embedder():
    """Load the sentence transformer model."""
    print(f"Loading model: {MODEL_NAME}...")
    return SentenceTransformer(MODEL_NAME)

def extract_keywords(docs, n_keywords=5):
    """Extract top keywords from a list of documents using TF-IDF."""
    if not docs:
        return []
    vectorizer = TfidfVectorizer(stop_words='english', max_features=100)
    try:
        tfidf_matrix = vectorizer.fit_transform(docs)
        scores = np.asarray(tfidf_matrix.sum(axis=0)).flatten()
        indices = scores.argsort()[-n_keywords:][::-1]
        features = vectorizer.get_feature_names_out()
        return [features[i] for i in indices]
    except:
        return ["topic"]

def save_state(centroids, cluster_names):
    """Save cluster centroids and names to disk."""
    STATE_DIR.mkdir(exist_ok=True)
    np.save(STATE_DIR / "centroids.npy", centroids)
    with open(STATE_DIR / "cluster_names.json", "w") as f:
        json.dump(cluster_names, f, indent=4)
    print(f"Saved cluster state to {STATE_DIR}")

def load_state():
    """Load cluster centroids and names from disk."""
    if not (STATE_DIR / "centroids.npy").exists():
        return None, None
    centroids = np.load(STATE_DIR / "centroids.npy")
    with open(STATE_DIR / "cluster_names.json", "r") as f:
        cluster_names = json.load(f)
    # Convert keys back to integers
    cluster_names = {int(k): v for k, v in cluster_names.items()}
    return centroids, cluster_names

def run_fit(input_file, text_column):
    """Initial clustering run."""
    print(f"Reading {input_file}...")
    if input_file.endswith('.xlsx'):
        df = pd.read_excel(input_file)
    else:
        df = pd.read_csv(input_file)

    if text_column not in df.columns:
        print(f"Error: Column '{text_column}' not found in {input_file}")
        return

    print("Normalizing text...")
    df['clean_text'] = df[text_column].apply(normalize_hinglish)
    valid_df = df[df['clean_text'] != ""].copy()

    model = get_embedder()
    print("Generating embeddings...")
    embeddings = model.encode(valid_df['clean_text'].tolist(), show_progress_bar=True)

    print("Reducing dimensions with UMAP...")
    reducer = umap.UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42)
    umap_embeddings = reducer.fit_transform(embeddings)

    print("Clustering with HDBSCAN...")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=MIN_CLUSTER_SIZE, metric='euclidean', cluster_selection_method='eom')
    cluster_labels = clusterer.fit_predict(umap_embeddings)

    valid_df['cluster'] = cluster_labels
    
    # Calculate centroids and extract names
    unique_labels = [l for l in np.unique(cluster_labels) if l != -1]
    centroids = []
    cluster_names = {}
    
    for label in unique_labels:
        mask = (cluster_labels == label)
        cluster_embeddings = embeddings[mask]
        centroid = cluster_embeddings.mean(axis=0)
        centroids.append(centroid)
        
        # Name the cluster
        cluster_docs = valid_df[valid_df['cluster'] == label]['clean_text'].tolist()
        keywords = extract_keywords(cluster_docs)
        cluster_names[int(label)] = " | ".join(keywords)

    save_state(np.array(centroids), cluster_names)
    
    # Add cluster names to dataframe
    valid_df['topic_name'] = valid_df['cluster'].map(lambda x: cluster_names.get(x, "Uncategorized"))
    
    # Visualization (using 2D UMAP for plot)
    print("Generating visualization...")
    vis_reducer = umap.UMAP(n_components=2, random_state=42)
    vis_coords = vis_reducer.fit_transform(embeddings)
    valid_df['x'] = vis_coords[:, 0]
    valid_df['y'] = vis_coords[:, 1]
    
    fig = px.scatter(valid_df, x='x', y='y', color='topic_name', hover_data=[text_column], title="Video Title Clusters")
    OUTPUT_DIR.mkdir(exist_ok=True)
    fig.write_html(OUTPUT_DIR / "cluster_visualization.html")
    valid_df.to_csv(OUTPUT_DIR / "clustered_data.csv", index=False)
    
    print(f"Results saved to {OUTPUT_DIR}")

def run_predict(input_file, text_column, threshold):
    """Incremental assignment to existing clusters."""
    centroids, cluster_names = load_state()
    if centroids is None:
        print("Error: No existing cluster state found. Run with --mode fit first.")
        return

    print(f"Reading {input_file}...")
    if input_file.endswith('.xlsx'):
        df = pd.read_excel(input_file)
    else:
        df = pd.read_csv(input_file)

    print("Normalizing text...")
    df['clean_text'] = df[text_column].apply(normalize_hinglish)
    
    model = get_embedder()
    print("Generating embeddings...")
    new_embeddings = model.encode(df['clean_text'].tolist(), show_progress_bar=True)

    print("Assigning to clusters based on similarity...")
    similarities = cosine_similarity(new_embeddings, centroids)
    
    max_sim_indices = np.argmax(similarities, axis=1)
    max_sim_values = np.max(similarities, axis=1)
    
    assigned_clusters = []
    assigned_topics = []
    
    for i, sim in enumerate(max_sim_values):
        if sim >= threshold:
            cluster_id = list(cluster_names.keys())[max_sim_indices[i]]
            assigned_clusters.append(cluster_id)
            assigned_topics.append(cluster_names[cluster_id])
        else:
            assigned_clusters.append(-1)
            assigned_topics.append("New/Potential Topic")

    df['assigned_cluster'] = assigned_clusters
    df['topic_name'] = assigned_topics
    df['similarity_score'] = max_sim_values

    OUTPUT_DIR.mkdir(exist_ok=True)
    df.to_csv(OUTPUT_DIR / "incremental_assignment_results.csv", index=False)
    
    flagged = df[df['assigned_cluster'] == -1]
    if not flagged.empty:
        flagged.to_csv(OUTPUT_DIR / "flagged_new_topics.csv", index=False)
        print(f"Flagged {len(flagged)} titles as new potential topics.")

    print(f"Assigned results saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clustering and naming social media content groups.")
    parser.add_argument("--mode", choices=["fit", "predict"], required=True, help="Mode: 'fit' for initial run, 'predict' for incremental.")
    parser.add_argument("--input", required=True, help="Path to input CSV or XLSX file.")
    parser.add_argument("--column", default="title", help="Name of the text column to cluster.")
    parser.add_argument("--threshold", type=float, default=SIMILARITY_THRESHOLD, help="Cosine similarity threshold for assignment.")
    
    args = parser.parse_args()
    
    if args.mode == "fit":
        run_fit(args.input, args.column)
    else:
        run_predict(args.input, args.column, args.threshold)
