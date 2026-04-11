import json
import os
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

import social_listening_bertopic as sl

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
    deps = sl.ensure_base_dependencies()
    BERTopic, KeyBERTInspired, MaximalMarginalRelevance, hdbscan_module, px, SentenceTransformer, umap_module = deps

    input_path = Path("Video Titles.xlsx")
    output_root = Path("outputs/expert_clustering")
    output_root.mkdir(parents=True, exist_ok=True)

    print("Step 1: Loading Data and Pre-bucketing...")
    df = sl.load_input(input_path, "xlsx", "title", "content")
    df = sl.dedupe_documents(df, enabled=True)
    
    embedder = sl.load_embedder(SentenceTransformer, "sentence-transformers/LaBSE")
    
    all_results = []
    next_global_id = 0
    
    buckets = df["coarse_bucket"].unique()
    print(f"Detected buckets: {buckets}")
    
    for bucket in buckets:
        bucket_df = df[df["coarse_bucket"] == bucket].copy()
        if len(bucket_df) < 15: 
            print(f"Skipping small bucket: {bucket}")
            bucket_df["topic"] = -1
            all_results.append(bucket_df)
            continue
            
        print(f"\nProcessing bucket: {bucket} ({len(bucket_df)} docs)")
        bucket_texts = bucket_df["ner_text"].tolist()
        bucket_embeddings = prepare_embeddings(embedder, bucket_texts, 32)
        
        # Target ~15 clusters per major bucket to get 80-90 total
        model = sl.build_topic_model(
            BERTopic=BERTopic,
            KeyBERTInspired=KeyBERTInspired,
            MaximalMarginalRelevance=MaximalMarginalRelevance,
            hdbscan_module=hdbscan_module,
            umap_module=umap_module,
            embedder=embedder,
            min_cluster_size=20, # Higher size for more stable clusters
            min_samples=8,
            umap_n_neighbors=15,
            umap_components=5,
            umap_min_dist=0.0,
            min_df=2,
            max_df=0.9,
            nr_topics=None,
            top_n_words=10,
            doc_count=len(bucket_df)
        )
        
        try:
            topics, _ = model.fit_transform(bucket_texts, bucket_embeddings)
        except ValueError:
            # Fallback for very small datasets or vectorizer issues
            model = sl.build_topic_model(
                BERTopic=BERTopic,
                KeyBERTInspired=KeyBERTInspired,
                MaximalMarginalRelevance=MaximalMarginalRelevance,
                hdbscan_module=hdbscan_module,
                umap_module=umap_module,
                embedder=embedder,
                min_cluster_size=15,
                min_samples=5,
                umap_n_neighbors=15,
                umap_components=5,
                umap_min_dist=0.0,
                min_df=1,
                max_df=1.0,
                nr_topics=None,
                top_n_words=10,
                doc_count=len(bucket_df)
            )
            topics, _ = model.fit_transform(bucket_texts, bucket_embeddings)
        
        topics = np.array(topics)
        unique_local = sorted([t for t in set(topics) if t != -1])
        mapping = {local: next_global_id + i for i, local in enumerate(unique_local)}
        mapping[-1] = -1
        
        global_topics = [mapping[t] for t in topics]
        bucket_df["topic"] = global_topics
        next_global_id += len(unique_local)
        all_results.append(bucket_df)

    final_df = pd.concat(all_results).reset_index(drop=True)
    
    # Step 3: Residual Sweep
    noise_df = final_df[final_df["topic"] == -1].copy()
    if len(noise_df) > 50:
        print(f"\nStep 3: Attempting recovery of {len(noise_df)} noise documents...")
        noise_texts = noise_df["ner_text"].tolist()
        noise_embeddings = prepare_embeddings(embedder, noise_texts, 32)
        
        noise_model = sl.build_topic_model(
            BERTopic=BERTopic,
            KeyBERTInspired=KeyBERTInspired,
            MaximalMarginalRelevance=MaximalMarginalRelevance,
            hdbscan_module=hdbscan_module,
            umap_module=umap_module,
            embedder=embedder,
            min_cluster_size=15,
            min_samples=5,
            umap_n_neighbors=10,
            umap_components=5,
            umap_min_dist=0.0,
            min_df=2,
            max_df=0.9,
            nr_topics=None,
            top_n_words=10,
            doc_count=len(noise_df)
        )
        
        noise_topics, _ = noise_model.fit_transform(noise_texts, noise_embeddings)
        noise_topics = np.array(noise_topics)
        
        unique_noise = sorted([t for t in set(noise_topics) if t != -1])
        noise_mapping = {local: next_global_id + i for i, local in enumerate(unique_noise)}
        noise_mapping[-1] = -1
        
        final_noise_topics = [noise_mapping[t] for t in noise_topics]
        final_df.loc[final_df["topic"] == -1, "topic"] = final_noise_topics
        next_global_id += len(unique_noise)

    # Step 4: Final Summary and JSON
    print("\nStep 4: Extracting keywords and generating hierarchy...")
    
    def get_ctfidf_keywords(df):
        valid_topics = df[df["topic"] != -1]
        if valid_topics.empty:
            return {}
            
        # Group documents by topic
        documents_per_topic = valid_topics.groupby("topic")["ner_text"].apply(lambda x: " ".join(x)).reset_index()
        
        vectorizer = CountVectorizer(stop_words="english", ngram_range=(1, 3))
        X = vectorizer.fit_transform(documents_per_topic["ner_text"])
        words = vectorizer.get_feature_names_out()
        
        # c-TF-IDF calculation
        tf = X.toarray()
        count = tf.sum(axis=1) # Length of each topic document
        avg_w = count.mean()
        df_word = (X > 0).sum(axis=0) # Frequency of each word across topics
        
        # Clean shapes for broadcasting
        df_word = np.asarray(df_word).flatten()
        idf = np.log(1 + avg_w / (df_word + 1))
        
        # Element-wise scaling: tf[M, N] * idf[N]
        ctfidf = tf * idf
        
        topic_keywords = {}
        for i, topic_id in enumerate(documents_per_topic["topic"]):
            top_indices = ctfidf[i].argsort()[-10:][::-1]
            topic_keywords[topic_id] = [words[idx] for idx in top_indices]
        return topic_keywords

    all_keywords = get_ctfidf_keywords(final_df)
    final_topic_counts = final_df["topic"].value_counts().to_dict()
    
    out_data = {"clusters": []}
    for t_id in sorted(all_keywords.keys()):
        kws = all_keywords[t_id]
        count = final_topic_counts.get(t_id, 0)
        samples = final_df[final_df["topic"] == t_id]["raw_text"].head(3).tolist()
        out_data["clusters"].append({
            "cluster_id": f"EX_{t_id}",
            "name": " / ".join(kws[:3]).title(),
            "doc_count": int(count),
            "keywords": kws,
            "sample_documents": samples,
            "parent_id": "root"
        })

    (output_root / "expert_hierarchy.json").write_text(json.dumps(out_data, indent=2, ensure_ascii=True))
    final_df.to_csv(output_root / "expert_clustered_documents.csv", index=False)
    
    print(f"\nFinal Statistics:")
    print(f"Total Clusters: {len(out_data['clusters'])}")
    print(f"Unassigned Noise: {final_topic_counts.get(-1, 0)} documents.")
    print(f"Results written to {output_root}")

if __name__ == "__main__":
    main()
