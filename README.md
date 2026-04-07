# Social Media Content Clustering

A high-fidelity clustering solution for social media content (specifically video titles) supporting English, Hindi, and Hinglish.

## Features
- **Multilingual Support**: Uses the `LaBSE` (Language-Agnostic BERT Sentence Embedding) model.
- **Hinglish Normalization**: Custom preprocessing to handle mixed-language text and Devanagari characters.
- **Advanced Clustering**: Combines UMAP for dimensionality reduction and HDBSCAN for density-based clustering.
- **Topic Naming**: Automatically generates topic names using TF-IDF keyword extraction.
- **Incremental Assignment**: Assign new content to existing topics based on semantic similarity.
- **Interactive Visualization**: Generates Plotly-based HTML scatter plots.

## Installation
Ensure you have a Python environment (3.10+) and install dependencies:
```bash
pip install pandas openpyxl sentence-transformers hdbscan umap-learn plotly scikit-learn
```

## Usage

### 1. Initial Clustering (Fit)
Process your main dataset to identify topics and save the cluster state.
```bash
python social_listening_hdbscan.py --mode fit --input "Video Titles.xlsx" --column "title"
```
**Outputs:**
- `outputs/clustered_data.csv`: Your data with assigned cluster IDs and topic names.
- `outputs/cluster_visualization.html`: Interactive 2D visualization of the clusters.
- `cluster_state/`: Directory containing saved centroids and metadata.

### 2. Incremental Assignment (Predict)
Assign new video titles to the topics identified in the first run.
```bash
python social_listening_hdbscan.py --mode predict --input "new_batch.csv" --column "title" --threshold 0.60
```
**Outputs:**
- `outputs/incremental_assignment_results.csv`: New titles with assigned topics and similarity scores.
- `outputs/flagged_new_topics.csv`: Titles that did not match any existing topic with sufficient similarity.

## Configuration
- `SIMILARITY_THRESHOLD`: Default `0.60`. Increase for stricter matching.
- `MIN_CLUSTER_SIZE`: Default `5`. Minimum number of items to form a new topic.
# multilingual-content-clusters
