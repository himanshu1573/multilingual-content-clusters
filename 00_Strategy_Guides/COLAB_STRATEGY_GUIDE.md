# 🧪 Hinglish News Clustering: Colab Strategy Guide

This document provides a consolidated overview of the **6 experimentation notebooks** developed for the Hinglish news clustering pipeline. Each notebook represents a specific phase of the project's evolution, from raw discovery to production-ready entity anchoring.

---

## 📂 Notebook Catalog

### 1. [Baseline Semantic BERTopic](file:///Users/himanshup/Clustering%20/colab_notebooks/01_Baseline_Semantic_BERTopic.ipynb)
*   **Methodology**: Pure unsupervised semantic discovery using **LaBSE** + **HDBSCAN**.
*   **Workflow**: Data Loading → Standard Hinglish Cleaning → Multilingual Embedding → Density-based Clustering.
*   **Best For**: Initial exploration of a 100% unknown dataset.
*   **Note**: Expect high noise (~25%) as HDBSCAN is strict about "pure" clusters.

### 2. [Multi-Strategy Comparison](file:///Users/himanshup/Clustering%20/colab_notebooks/02_MultiStrategy_Comparison.ipynb)
*   **Methodology**: Benchmarking 3 different paths: (A) Tuned Baseline, (B) NER-Normalized, and (C) Hybrid Bucketed.
*   **Workflow**: Parallel runs on the same dataset → Metric extraction (Silhouette, Noise Ratio, Dominance).
*   **Best For**: Verifying which architectural approach provides the best "Signal-to-Noise" ratio for your specific content.
*   **Note**: Useful for stakeholder presentations to justify model selection.

### 3. [Multi-Pass Noise Recovery](file:///Users/himanshup/Clustering%20/colab_notebooks/03_MultiPass_Noise_Recovery.ipynb)
*   **Methodology**: A **7-pass architecture** that uses iterative refinement.
*   **Workflow**: Strict Clustering → Deduplication → **Cosine Outlier Reduction** (threshold=0.65) → Hierarchy Building → Granularity Optimization.
*   **Best For**: Achieving high data coverage (reducing noise) while maintaining distinct sub-topic boundaries.
*   **Note**: Produces a hierarchical tree view for nested content navigation.

### 4. [Expert Bucketed Engine](file:///Users/himanshup/Clustering%20/colab_notebooks/04_Expert_Bucketed_Engine.ipynb)
*   **Methodology**: **Divide-and-Conquer**. Pre-groups titles into domains (Politics, Geopolitics, etc.) before clustering.
*   **Workflow**: Domain Classification (Regex/Zero-Shot) → Independent BERTopic runs per bucket → **On-the-fly LLM Labeling** (Qwen/Ollama).
*   **Best For**: Large-scale datasets where small, important topics might get drowned out by major news stories (like the Iran war).
*   **Note**: The "Gold Standard" for structured news dashboards.

### 5. [Entity-Anchored Guided Clustering](file:///Users/himanshup/Clustering%20/colab_notebooks/05_Entity_Anchored_Guided.ipynb)
*   **Methodology**: **Guided Anchor Clustering**. Prioritizes People/Places as the "seeds" of topics.
*   **Workflow**: NER Extraction → Top-N Entity Frequency Mapping → **Guided Topic Seeding** → Hard Outlier Reduction (Strategy: Embeddings).
*   **Best For**: **Production Search APIs**. Ensures topics are anchored to famous entities (Modi, Trump) and achieves **Zero Noise**.
*   **Note**: Every single document is forced into a category, making it perfect for search indexing.

### 6. [NER & Entity Frequency Analysis](file:///Users/himanshup/Clustering%20/colab_notebooks/06_NER_Entity_Extraction.ipynb)
*   **Methodology**: Pure Entity Intelligence.
*   **Workflow**: Named Entity Recognition (WikiNeural) → Entity Deduplication → Frequency Distribution Mapping.
*   **Best For**: Understanding the "Who, What, Where" of your data before you even start clustering.
*   **Note**: Essential for generating the seed lists used in Notebook 05.

---

## 🛠️ Technical Flow Summary (Standard)

Regardless of the strategy, the core processing engine follows this "Golden Flow" we've refined:

1.  **Hinglish Normalization**: Stripping hashtags/channel codes while protecting **Devanagari** (UTF-8).
2.  **Multilingual Vectorization**: Using **LaBSE** to bridge the gap between English and Hindi tokens.
3.  **Dimensionality Reduction**: Using **UMAP** for density-based grouping.
4.  **Semantic Weighting**: Using **c-TF-IDF** to find the most "important" words for each news cluster.
5.  **LLM Enrichment**: Using **Qwen2.5:3b** to turn raw keywords into professional news headlines.
