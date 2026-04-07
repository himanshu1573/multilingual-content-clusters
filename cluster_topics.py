import argparse
import html
import json
import re
import string
from collections import Counter
from pathlib import Path

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer


MONTHS = {
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
}

DOMAIN_STOPS = {
    "rahulkanwal",
    "rahul",
    "kanwal",
    "twitter",
    "tweet",
    "news",
    "amp",
    "com",
    "co",
    "https",
    "http",
    "www",
    "xcom",
    "breaking",
    "ndtv",
    "news",
    "follow",
    "click",
    "latest",
    "updates",
    "watch",
    "youtube",
    "channel",
    "playlist",
    "facebook",
    "instagram",
    "whatsapp",
    "apps",
    "mobile",
    "videos",
    "video",
    "subscribe",
    "shows",
    "exclusive",
    "live",
    "shorts",
    "hindi",
    "english",
    "media",
    "post",
    "posts",
    "social",
    "share",
    "list",
    "link",
    "links",
    "today",
    "tonight",
    "latestupdates",
    "editorinchief",
    "ceo",
    "inchief",
}

STOPS = set(ENGLISH_STOP_WORDS) | MONTHS | DOMAIN_STOPS

AC_CODE_RE = re.compile(r"AC/\d{1,4}/\d{1,4}", re.IGNORECASE)
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#(\w+)")
NON_ALPHA_RE = re.compile(r"[^a-z\s]+")
SPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    text = html.unescape(text or "")
    text = AC_CODE_RE.sub(" ", text)
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = HASHTAG_RE.sub(r" \1 ", text)
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = NON_ALPHA_RE.sub(" ", text)
    words = [word for word in text.split() if word not in STOPS and len(word) > 2]
    return SPACE_RE.sub(" ", " ".join(words)).strip()


def load_posts(input_path: Path) -> list[dict]:
    payload = json.loads(input_path.read_text())
    posts = payload.get("posts", [])
    if not isinstance(posts, list):
        raise ValueError("Expected JSON payload to contain a list under 'posts'.")
    return posts


def cluster_posts(
    posts: list[dict],
    n_clusters: int,
    max_features: int,
    min_df: int,
    max_df: float,
    top_terms: int,
    samples_per_cluster: int,
    dedupe: bool,
) -> dict:
    if dedupe:
        seen = set()
        filtered_posts = []
        for post in posts:
            content = (post.get("content") or "").strip()
            if content in seen:
                continue
            seen.add(content)
            filtered_posts.append(post)
    else:
        filtered_posts = posts

    docs = [post.get("content") or "" for post in filtered_posts]
    cleaned_docs = [clean_text(doc) for doc in docs]

    kept_rows = []
    kept_docs = []
    for idx, cleaned in enumerate(cleaned_docs):
        if cleaned:
            kept_rows.append(idx)
            kept_docs.append(cleaned)

    if len(kept_docs) < n_clusters:
        raise ValueError(
            f"Need at least {n_clusters} non-empty cleaned documents, found {len(kept_docs)}."
        )

    vectorizer = TfidfVectorizer(
        lowercase=True,
        max_features=max_features,
        max_df=max_df,
        min_df=min_df,
        ngram_range=(1, 3),
        stop_words="english",
    )
    vectors = vectorizer.fit_transform(kept_docs)

    model = KMeans(
        n_clusters=n_clusters,
        init="k-means++",
        max_iter=100,
        n_init=10,
        random_state=42,
    )
    labels = model.fit_predict(vectors)

    order_centroids = model.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names_out()

    clusters = []
    label_counts = Counter(labels)
    for cluster_id in range(n_clusters):
        top_term_indices = order_centroids[cluster_id, :top_terms]
        cluster_terms = [terms[idx] for idx in top_term_indices]

        cluster_doc_ids = [i for i, label in enumerate(labels) if label == cluster_id]
        samples = []
        for local_idx in cluster_doc_ids[:samples_per_cluster]:
            original_idx = kept_rows[local_idx]
            post = filtered_posts[original_idx]
            samples.append(
                {
                    "post_id": post.get("id"),
                    "platform": post.get("platform"),
                    "posted_at": post.get("posted_at"),
                    "content": (post.get("content") or "").strip(),
                }
            )

        clusters.append(
            {
                "cluster_id": cluster_id,
                "document_count": label_counts[cluster_id],
                "top_terms": cluster_terms,
                "sample_posts": samples,
            }
        )

    return {
        "summary": {
            "total_posts": len(posts),
            "deduped_posts": len(filtered_posts),
            "non_empty_cleaned_posts": len(kept_docs),
            "n_clusters": n_clusters,
            "max_features": max_features,
            "min_df": min_df,
            "max_df": max_df,
            "top_terms_per_cluster": top_terms,
            "samples_per_cluster": samples_per_cluster,
            "vocabulary_size": len(terms),
            "feature_matrix_shape": list(vectors.shape),
            "inertia": float(model.inertia_),
        },
        "clusters": clusters,
    }


def write_report(report: dict, output_path: Path) -> None:
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=True))


def print_digest(report: dict) -> None:
    summary = report["summary"]
    print("TF-IDF + K-Means Topic Digest")
    print("=" * 32)
    print(f"Total posts: {summary['total_posts']}")
    print(f"Deduped posts used: {summary['deduped_posts']}")
    print(f"Usable cleaned posts: {summary['non_empty_cleaned_posts']}")
    print(f"Clusters: {summary['n_clusters']}")
    print(f"Vocabulary size: {summary['vocabulary_size']}")
    print(f"Feature matrix shape: {tuple(summary['feature_matrix_shape'])}")
    print(f"Inertia: {summary['inertia']:.4f}")
    print()

    for cluster in report["clusters"]:
        print(f"Cluster {cluster['cluster_id']} ({cluster['document_count']} posts)")
        print("Top terms:", ", ".join(cluster["top_terms"]))
        for idx, sample in enumerate(cluster["sample_posts"], start=1):
            snippet = SPACE_RE.sub(" ", sample["content"])[:220]
            print(f"Sample {idx}: {snippet}")
        print("-" * 60)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cluster post text with TF-IDF + K-Means.")
    parser.add_argument(
        "--input",
        default="Rahul_Kanwal_posts_20260407_121843.json",
        help="Path to the JSON export file.",
    )
    parser.add_argument(
        "--output",
        default="topic_clusters_report.json",
        help="Path to write the cluster report JSON.",
    )
    parser.add_argument("--clusters", type=int, default=20, help="Number of K-Means clusters.")
    parser.add_argument(
        "--max-features",
        type=int,
        default=300,
        help="Maximum TF-IDF vocabulary size.",
    )
    parser.add_argument(
        "--min-df",
        type=int,
        default=3,
        help="Ignore terms appearing in fewer than this many posts.",
    )
    parser.add_argument(
        "--max-df",
        type=float,
        default=0.8,
        help="Ignore terms appearing in more than this share of posts.",
    )
    parser.add_argument(
        "--top-terms",
        type=int,
        default=10,
        help="Number of top terms to show per cluster.",
    )
    parser.add_argument(
        "--samples-per-cluster",
        type=int,
        default=3,
        help="Number of sample posts to show per cluster.",
    )
    parser.add_argument(
        "--no-dedupe",
        action="store_true",
        help="Keep exact duplicate post texts instead of removing them before clustering.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    posts = load_posts(input_path)
    report = cluster_posts(
        posts=posts,
        n_clusters=args.clusters,
        max_features=args.max_features,
        min_df=args.min_df,
        max_df=args.max_df,
        top_terms=args.top_terms,
        samples_per_cluster=args.samples_per_cluster,
        dedupe=not args.no_dedupe,
    )
    write_report(report, output_path)
    print_digest(report)


if __name__ == "__main__":
    main()
