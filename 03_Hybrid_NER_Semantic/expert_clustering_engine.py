import json
import os
import requests
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

import social_listening_bertopic as sl

# ELABORATIVE NAMES MAPPING
ELABORATIVE_NAMES = {
    "geopolitical_conflict_and_international_relations": "Geopolitical Relations & International Conflict",
    "national_electoral_politics_and_governance": "National & State Electoral Politics",
    "legislative_affairs_and_public_policy": "Legislative Affairs & Public Policy",
    "regional_public_safety_and_security_incidents": "Regional Public Safety & Security Incidents",
    "economic_trends_and_infrastructure": "Economic Trends & Socio-Economic Infrastructure",
    "culture_media_and_digital_lifestyles": "Cultural Trends, Media & Digital Lifestyle",
    "general_misc": "General Informational & Miscellaneous News",
    "residual_recovery": "Cross-Bucket Residual Topic Recovery"
}

def generate_llm_label(parent_bucket, keywords, docs):
    """Generate a specific label on the go using the parent context."""
    prompt = f"""You are a professional news analyst. Create a clear, specific topic label within the category: "{parent_bucket}"
Format: Entity/Main Concept + Action/Detail (Context in parentheses)
Examples:
- US Rescue Mission In Iran (Trump, Pilot)
- West Bengal Election Campaign 2026 (BJP Opinion Poll)
- Seaplane Trial on Ganga (Uttarakhand)

Return ONLY the label. Do not use quotes or phrases like "Label:".

Keywords: {', '.join(keywords)}
Sample documents:
1. {docs[0]}
2. {docs[1] if len(docs) > 1 else ''}

Label:"""
    
    try:
        response = requests.post(
            "http://127.0.0.1:11434/api/generate",
            json={
                "model": "qwen2.5:3b",
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 40
                }
            },
            timeout=30
        )
        if response.status_code == 200:
            return response.json().get("response", "").strip().replace("Label:", "").strip()
    except Exception as e:
        print(f"  Warning: LLM label failed: {e}")
        return " / ".join(keywords[:3]).title()

def prepare_embeddings(embedder: Any, texts: list[str], batch_size: int) -> np.ndarray:
    return np.asarray(
        embedder.encode(
            texts,
            show_progress_bar=True,
            batch_size=batch_size,
            normalize_embeddings=True,
        )
    )

def extract_keywords(docs_list, topic_ids):
    """Local c-TF-IDF for a set of docs and their labels."""
    df = pd.DataFrame({"text": docs_list, "topic": topic_ids})
    df = df[df["topic"] != -1]
    if df.empty: return {}
    
    docs_per_topic = df.groupby("topic")["text"].apply(lambda x: " ".join(x)).reset_index()
    vectorizer = CountVectorizer(stop_words="english", ngram_range=(1, 3))
    X = vectorizer.fit_transform(docs_per_topic["text"])
    words = vectorizer.get_feature_names_out()
    
    tf = X.toarray()
    count = tf.sum(axis=1)
    avg_w = count.mean()
    df_word = (X > 0).sum(axis=0)
    df_word = np.asarray(df_word).flatten()
    idf = np.log(1 + avg_w / (df_word + 1))
    ctfidf = tf * idf
    
    res = {}
    for i, t_id in enumerate(docs_per_topic["topic"]):
        top_indices = ctfidf[i].argsort()[-10:][::-1]
        res[t_id] = [words[idx] for idx in top_indices]
    return res

def main():
    deps = sl.ensure_base_dependencies()
    BERTopic, KeyBERTInspired, MaximalMarginalRelevance, hdbscan_module, px, SentenceTransformer, umap_module = deps

    input_path = Path("Video Titles.xlsx")
    output_root = Path("outputs/expert_engine")
    output_root.mkdir(parents=True, exist_ok=True)

    print("Step 1: Loading Data and Preparing Embeddings...")
    df = sl.load_input(input_path, "xlsx", "title", "content")
    df = sl.dedupe_documents(df, enabled=True)
    embedder = sl.load_embedder(SentenceTransformer, "sentence-transformers/LaBSE")
    
    hierarchy = {}
    all_results = []
    next_global_id = 0
    
    buckets = df["coarse_bucket"].unique()
    
    for bucket_id in buckets:
        parent_name = ELABORATIVE_NAMES.get(bucket_id, bucket_id)
        bucket_df = df[df["coarse_bucket"] == bucket_id].copy()
        
        if len(bucket_df) < 15:
            print(f"Skipping tiny bucket: {parent_name}")
            bucket_df["topic"] = -1
            all_results.append(bucket_df)
            continue
            
        print(f"\n>>> Processing Bucket: {parent_name} ({len(bucket_df)} docs)")
        bucket_texts = bucket_df["ner_text"].tolist()
        bucket_embeddings = prepare_embeddings(embedder, bucket_texts, 32)
        
        model = sl.build_topic_model(
            BERTopic=BERTopic, KeyBERTInspired=KeyBERTInspired,
            MaximalMarginalRelevance=MaximalMarginalRelevance,
            hdbscan_module=hdbscan_module, umap_module=umap_module,
            embedder=embedder, min_cluster_size=18, min_samples=7,
            umap_n_neighbors=15, umap_components=5, umap_min_dist=0.0,
            min_df=2, max_df=0.9, nr_topics=None, top_n_words=10,
            doc_count=len(bucket_df)
        )
        
        try:
            topics, _ = model.fit_transform(bucket_texts, bucket_embeddings)
        except:
            print(f"  Retrying with safer params for {parent_name}...")
            model = sl.build_topic_model(
                BERTopic=BERTopic, KeyBERTInspired=KeyBERTInspired,
                MaximalMarginalRelevance=MaximalMarginalRelevance,
                hdbscan_module=hdbscan_module, umap_module=umap_module,
                embedder=embedder, min_cluster_size=12, min_samples=5,
                umap_n_neighbors=15, umap_components=5, umap_min_dist=0.0,
                min_df=1, max_df=1.0, nr_topics=None, top_n_words=10,
                doc_count=len(bucket_df)
            )
            topics, _ = model.fit_transform(bucket_texts, bucket_embeddings)
            
        topics = np.array(topics)
        unique_local = sorted([t for t in set(topics) if t != -1])
        
        # Immediate Labeling for this Bucket
        print(f"  Generating labels for {len(unique_local)} sub-topics...")
        keywords = extract_keywords(bucket_texts, topics)
        
        sub_topics = []
        mapping = {-1: -1}
        
        for i, local_id in enumerate(unique_local):
            global_id = next_global_id + i
            mapping[local_id] = global_id
            
            kws = keywords.get(local_id, ["news"])
            # Find sample docs
            indices = np.where(topics == local_id)[0][:2]
            samples = [bucket_df.iloc[idx]["raw_text"] for idx in indices]
            
            # CALL LLM ON THE GO
            refined_name = generate_llm_label(parent_name, kws, samples)
            print(f"    [{global_id}] -> {refined_name}")
            
            sub_topics.append({
                "id": int(global_id),
                "name": refined_name,
                "count": int((topics == local_id).sum()),
                "keywords": kws,
                "samples": samples
            })
            
        hierarchy[parent_name] = {
            "bucket_id": bucket_id,
            "total_docs": len(bucket_df),
            "sub_topics": sub_topics
        }
        
        bucket_df["topic"] = [mapping[t] for t in topics]
        next_global_id += len(unique_local)
        all_results.append(bucket_df)

    # FINAL EXPORT
    final_df = pd.concat(all_results).reset_index(drop=True)
    
    # Residual Sweep Logic (Simplified for the Engine)
    noise_count = (final_df["topic"] == -1).sum()
    print(f"\nFinal Noise Level: {noise_count} documents.")
    
    print(f"Saving Expert Results to {output_root}...")
    (output_root / "expert_nested_hierarchy.json").write_text(json.dumps(hierarchy, indent=2, ensure_ascii=False))
    final_df.to_csv(output_root / "expert_engine_documents.csv", index=False)
    
    print("\nExpert Clustering Engine Execution Complete.")

if __name__ == "__main__":
    main()
