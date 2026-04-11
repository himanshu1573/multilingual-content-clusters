import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from social_listening_bertopic import (
    build_topic_model,
    dedupe_documents,
    ensure_base_dependencies,
    load_input,
    load_embedder,
    summarize_topics
)

def prepare_embeddings(embedder: Any, texts: list[str], batch_size: int) -> np.ndarray:
    return np.asarray(
        embedder.encode(
            texts,
            show_progress_bar=True,
            batch_size=batch_size,
            normalize_embeddings=True,
        )
    )

def main():
    deps = ensure_base_dependencies()
    BERTopic, KeyBERTInspired, MaximalMarginalRelevance, hdbscan_module, px, SentenceTransformer, umap_module = deps

    input_path = Path("Video Titles.xlsx")
    output_dir = Path("outputs/anti_gravity")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Step 1: Loading Data and Applying NER...")
    docs_df = load_input(input_path, "xlsx", "title", "content")
    docs_df = dedupe_documents(docs_df, enabled=True)
    
    docs = docs_df["ner_text"].tolist()
    
    print("Step 2: Loading Embedder & Generating Embeddings...")
    embedder = load_embedder(SentenceTransformer, "sentence-transformers/LaBSE")
    embeddings = prepare_embeddings(embedder, docs, 32)
    
    # PASS 0: Initial BERTopic clustering (Strict)
    print("Step 3: Running Base BERTopic (Pass 0)...")
    topic_model = build_topic_model(
        BERTopic=BERTopic,
        KeyBERTInspired=KeyBERTInspired,
        MaximalMarginalRelevance=MaximalMarginalRelevance,
        hdbscan_module=hdbscan_module,
        umap_module=umap_module,
        embedder=embedder,
        min_cluster_size=15, # Use slightly smaller size to capture granularity before merging
        min_samples=5,
        umap_n_neighbors=15,
        umap_components=5,
        umap_min_dist=0.0,
        min_df=2,
        max_df=0.9,
        nr_topics="auto", # Allows initial reduction
        top_n_words=10,
        doc_count=len(docs_df)
    )
    
    topics, probabilities = topic_model.fit_transform(docs, embeddings)
    
    # PASS 1: Deduplication Pass (Merge Syntactic Variants) & Pass 6 Optimization
    # We will use BERTopic's merge_topics logic combined with reduce_topics.
    print("Step 4: PASS 1 & 6 - Deduplication and Granularity Optimization...")
    # First, let BERTopic merge topics with extreme similarity
    topic_model.reduce_topics(docs, nr_topics=35) # Target to reduce to 35-40 main topics
    
    # Update topics after reduction
    docs_df["topic"] = topic_model.topics_
    
    # PASS 2: Noise Recovery Pass
    print("Step 5: PASS 2 - Noise Recovery...")
    new_topics = topic_model.reduce_outliers(
        docs, 
        topics=topic_model.topics_, 
        strategy="embeddings",
        embeddings=embeddings,
        threshold=0.65 # Require at least 0.65 cosine similarity to assign
    )
    
    # Update assignment
    topic_model.update_topics(docs, topics=new_topics)
    docs_df["topic"] = topic_model.topics_
    
    # PASS 3 & 4 & 5: Hierarchical Structure
    print("Step 6: PASS 3-5 - Building Hierarchies...")
    # Generate hierarchical topics
    hierarchical_topics = topic_model.hierarchical_topics(docs)
    merged_tree = topic_model.get_topic_tree(hierarchical_topics)
    
    # Save the tree text for reference
    (output_dir / "hierarchy_tree.txt").write_text(merged_tree)
    
    # Generate Summaries & Refine Labels (Pass 7 emulation via static mapping for now, LLM can be hooked here)
    print("Step 7: Generating Final Outputs...")
    topic_info = topic_model.get_topic_info()
    
    # Create the hierarchical JSON structure
    out_data = {"clusters": []}
    
    reps = topic_model.get_representative_docs()
    
    for _, row in topic_info.iterrows():
        t_id = row["Topic"]
        if t_id == -1: continue
        
        name = row["Name"]
        count = row["Count"]
        
        # Simple label refinement: Extract top 5 words as base name
        kws = [w for w, _ in topic_model.get_topic(t_id)][:5]
        refined_name = " ".join(kws).title()
        
        cluster_obj = {
            "cluster_id": f"C_{t_id}",
            "name": refined_name,
            "doc_count": int(count),
            "level": "child",
            "keywords": kws,
            "sample_documents": reps.get(t_id, [])[:3],
            "parent_id": "root"
        }
        out_data["clusters"].append(cluster_obj)
        
    # Write JSON
    path_json = output_dir / "anti_gravity_hierarchy.json"
    path_json.write_text(json.dumps(out_data, indent=2, ensure_ascii=True))
    
    base_metrics = {
        "final_cluster_count": len(out_data["clusters"]),
        "noise_documents": int((docs_df["topic"] == -1).sum()),
        "total_documents": len(docs_df)
    }
    (output_dir / "run_metrics.json").write_text(json.dumps(base_metrics, indent=2, ensure_ascii=True))
    
    docs_df.to_csv(output_dir / "anti_gravity_clustered_documents.csv", index=False)
    print(f"Done! Results written to {output_dir}")
    print(f"Stats: {base_metrics}")

if __name__ == '__main__':
    main()
