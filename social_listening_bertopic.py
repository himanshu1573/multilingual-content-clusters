import argparse
import hashlib
import html
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer
from sklearn.metrics import calinski_harabasz_score, silhouette_score


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
    "abp",
    "amp",
    "breaking",
    "breakingnews",
    "channel",
    "click",
    "com",
    "editorinchief",
    "exclusive",
    "facebook",
    "follow",
    "hindi",
    "http",
    "https",
    "inchief",
    "instagram",
    "latest",
    "live",
    "mobile",
    "ndtv",
    "news",
    "playlist",
    "rahul",
    "rahulkanwal",
    "share",
    "shorts",
    "shots",
    "shows",
    "subscribe",
    "today",
    "tonight",
    "tweet",
    "updates",
    "video",
    "videos",
    "watch",
    "whatsapp",
    "www",
    "xcom",
    "viral",
    "viralnews",
    "viralvideo",
    "ytshorts",
    "youtube",
}

EXPERIMENT_STOPWORDS = {
    "aajtak",
    "aajtakdigital",
    "ndtvindia",
    "timesnownavbharat",
    "tv9",
    "tv9d",
    "news18",
    "n18s",
    "topnews",
    "parliamentnews",
    "shortvideo",
    "shortsvideo",
    "viralshorts",
    "political",
}

STOPS = set(ENGLISH_STOP_WORDS) | MONTHS | DOMAIN_STOPS | EXPERIMENT_STOPWORDS
TEXT_CANDIDATES = ("content", "text", "post", "title", "description", "caption")

CANONICAL_PATTERNS = [
    (re.compile(r"\brahul\s+gandhi\b|\brahulgandhi\b", re.IGNORECASE), " rahul_gandhi "),
    (
        re.compile(
            r"\bhimanta\s+biswa\s+sarma\b|\bhimantabiswasarma\b|\bhemantabiswasarma\b",
            re.IGNORECASE,
        ),
        " himanta_biswa_sarma ",
    ),
    (re.compile(r"\bpriyanka\s+gandhi\b|\bpriyankagandhi\b", re.IGNORECASE), " priyanka_gandhi "),
    (re.compile(r"\bakhilesh\s+yadav\b|\bakhileshyadav\b", re.IGNORECASE), " akhilesh_yadav "),
    (re.compile(r"\bbhagwant\s+mann\b", re.IGNORECASE), " bhagwant_mann "),
    (re.compile(r"\bravi\s+kishan\b|\bravikishan\b", re.IGNORECASE), " ravi_kishan "),
    (re.compile(r"\brekha\s+gupta\b", re.IGNORECASE), " rekha_gupta "),
    (re.compile(r"\bwomen\s+reservation\s+bill\b", re.IGNORECASE), " women_reservation_bill "),
    (re.compile(r"\bmahila\s+aarakshan\b|\bमहिला\s+आरक्षण\b"), " women_reservation_bill "),
    (re.compile(r"\bdelhi\s+assembly\b", re.IGNORECASE), " delhi_assembly "),
    (re.compile(r"\bpress\s+conference\b", re.IGNORECASE), " press_conference "),
    (re.compile(r"\bbjp\b", re.IGNORECASE), " bjp "),
    (re.compile(r"\bcongress\b", re.IGNORECASE), " congress "),
    (re.compile(r"\baap\b", re.IGNORECASE), " aap "),
    (re.compile(r"\bassam\b", re.IGNORECASE), " assam "),
    (re.compile(r"\bdelhi\b", re.IGNORECASE), " delhi "),
]

COARSE_BUCKET_RULES = {
    "international_conflict": {
        "iran",
        "israel",
        "trump",
        "middleeastwar",
        "hormuz",
        "tehran",
        "america",
        "usa",
        "war",
    },
    "elections_politics": {
        "assam",
        "election",
        "elections",
        "poll",
        "polls",
        "rahul_gandhi",
        "himanta_biswa_sarma",
        "bjp",
        "congress",
        "akhilesh_yadav",
        "priyanka_gandhi",
        "bhagwant_mann",
    },
    "parliament_policy": {
        "women_reservation_bill",
        "reservation",
        "parliament",
        "lok",
        "sabha",
        "bill",
        "policy",
        "aarakshan",
        "आरक्षण",
    },
    "local_security": {
        "bulldozer",
        "mumbai",
        "delhi_assembly",
        "security",
        "breach",
        "crime",
        "police",
        "unnao",
    },
    "entertainment_misc": {
        "virat",
        "anushka",
        "dhurandhar",
        "celeb",
        "movie",
        "actor",
        "girl",
        "vada",
    },
}

AC_CODE_RE = re.compile(r"AC/\d{1,4}/\d{1,4}", re.IGNORECASE)
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#(\w+)")
SEPARATOR_RE = re.compile(r"[-/|:+]+")
CAMEL_SPLIT_RE = re.compile(r"(?<=[a-z])(?=[A-Z])")
NON_WORD_RE = re.compile(r"[^\w\s\u0900-\u097F]")
MULTISPACE_RE = re.compile(r"\s+")


def ensure_base_dependencies() -> tuple[Any, Any, Any, Any, Any, Any]:
    try:
        from bertopic import BERTopic
        from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance
        import hdbscan
        import plotly.express as px
        from sentence_transformers import SentenceTransformer
        import umap
    except ImportError as exc:
        raise SystemExit(
            "Missing BERTopic dependencies. Install requirements.txt and rerun. "
            f"Original error: {exc}"
        ) from exc

    return BERTopic, KeyBERTInspired, MaximalMarginalRelevance, hdbscan, px, SentenceTransformer, umap


def load_embedder(sentence_transformer_cls: Any, model_name: str) -> Any:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    try:
        return sentence_transformer_cls(model_name, local_files_only=True)
    except TypeError:
        return sentence_transformer_cls(model_name)


def parse_nr_topics(value: str | None) -> int | str | None:
    if value is None:
        return None
    lowered = value.strip().lower()
    if lowered in {"none", ""}:
        return None
    if lowered == "auto":
        return "auto"
    return int(lowered)


def detect_input_type(path: Path, input_type: str) -> str:
    if input_type != "auto":
        return input_type

    suffix = path.suffix.lower()
    if suffix == ".json":
        return "json"
    if suffix == ".csv":
        return "csv"
    if suffix in {".xlsx", ".xls"}:
        return "xlsx"
    raise ValueError(f"Could not infer input type from suffix '{suffix}'.")


def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    text = AC_CODE_RE.sub(" ", text)
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = HASHTAG_RE.sub(r" \1 ", text)
    text = CAMEL_SPLIT_RE.sub(" ", text)
    text = SEPARATOR_RE.sub(" ", text)
    text = text.lower().strip()
    text = NON_WORD_RE.sub(" ", text)

    tokens = []
    for token in text.split():
        if len(token) <= 2:
            continue
        if token.isascii() and token in STOPS:
            continue
        tokens.append(token)
    return MULTISPACE_RE.sub(" ", " ".join(tokens)).strip()


def normalize_entities(text: str) -> str:
    value = html.unescape(text or "")
    for pattern, replacement in CANONICAL_PATTERNS:
        value = pattern.sub(replacement, value)
    return value


def build_variant_text(raw_text: str, variant: str) -> str:
    source = raw_text
    if variant == "ner":
        source = normalize_entities(source)
    return normalize_text(source)


def assign_coarse_bucket(clean_text: str) -> str:
    tokens = set(clean_text.split())
    best_bucket = "general_misc"
    best_score = 0
    for bucket, rules in COARSE_BUCKET_RULES.items():
        score = len(tokens & rules)
        if score > best_score:
            best_bucket = bucket
            best_score = score
    return best_bucket


def choose_text_column(df: pd.DataFrame, requested: str | None) -> str:
    if requested and requested in df.columns:
        return requested

    for candidate in TEXT_CANDIDATES:
        if candidate in df.columns:
            return candidate

    object_columns = [col for col in df.columns if df[col].dtype == object]
    if object_columns:
        return object_columns[0]

    raise ValueError("Could not detect a text column in the tabular input.")


def load_json_posts(path: Path, json_text_key: str) -> pd.DataFrame:
    payload = json.loads(path.read_text())
    if isinstance(payload, dict) and isinstance(payload.get("posts"), list):
        posts = payload["posts"]
    elif isinstance(payload, list):
        posts = payload
    else:
        raise ValueError("Expected a list of records or a dict with a 'posts' list.")

    rows = []
    for idx, post in enumerate(posts):
        if not isinstance(post, dict):
            continue
        rows.append(
            {
                "source_row": idx,
                "source_id": post.get("id", idx),
                "platform": post.get("platform"),
                "posted_at": post.get("posted_at"),
                "author_username": (post.get("author") or {}).get("username"),
                "author_display_name": (post.get("author") or {}).get("display_name"),
                "raw_text": str(post.get(json_text_key) or "").strip(),
            }
        )

    return pd.DataFrame(rows)


def load_tabular(path: Path, input_type: str, column: str | None) -> pd.DataFrame:
    if input_type == "csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path)

    text_column = choose_text_column(df, column)
    out = df.copy()
    out["raw_text"] = out[text_column].fillna("").astype(str).str.strip()
    out["source_row"] = np.arange(len(out))
    out["source_id"] = out.index.astype(str)
    return out


def load_input(path: Path, input_type: str, column: str | None, json_text_key: str) -> pd.DataFrame:
    resolved_type = detect_input_type(path, input_type)
    if resolved_type == "json":
        df = load_json_posts(path, json_text_key)
    elif resolved_type in {"csv", "xlsx"}:
        df = load_tabular(path, resolved_type, column)
    else:
        raise ValueError(f"Unsupported input type '{resolved_type}'.")

    if "raw_text" not in df.columns:
        raise ValueError("Input loader failed to create a 'raw_text' column.")

    df["raw_text"] = df["raw_text"].fillna("").astype(str).str.strip()
    df = df[df["raw_text"] != ""].copy()
    df["clean_text"] = df["raw_text"].map(lambda text: build_variant_text(text, "clean"))
    df["ner_text"] = df["raw_text"].map(lambda text: build_variant_text(text, "ner"))
    df["coarse_bucket"] = df["ner_text"].map(assign_coarse_bucket)
    df = df[df["clean_text"] != ""].reset_index(drop=True)
    return df


def dedupe_documents(df: pd.DataFrame, enabled: bool) -> pd.DataFrame:
    if not enabled:
        return df.reset_index(drop=True)
    subset = ["clean_text"]
    if "ner_text" in df.columns:
        subset.append("ner_text")
    return df.drop_duplicates(subset=subset).reset_index(drop=True)


def select_text_column(df: pd.DataFrame, text_variant: str) -> str:
    return "ner_text" if text_variant == "ner" else "clean_text"


def build_keyword_label(keywords: list[str], fallback_id: int) -> str:
    if not keywords:
        return f"Topic {fallback_id}"
    return " / ".join(word.replace("_", " ") for word in keywords[:3]).title()


def topic_signature(keywords: list[str], representative_docs: list[str]) -> str:
    payload = json.dumps(
        {
            "keywords": keywords[:10],
            "documents": representative_docs[:3],
        },
        ensure_ascii=True,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_label_cache(cache_path: Path) -> dict[str, Any]:
    if not cache_path.exists():
        return {"model_name": None, "topics": {}}
    return json.loads(cache_path.read_text())


def save_label_cache(cache_path: Path, cache: dict[str, Any]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, indent=2, ensure_ascii=True))


def clean_generated_label(label: str, fallback: str) -> str:
    value = label.strip().strip('"').strip("'")
    if "label:" in value.lower():
        value = value.split(":", 1)[-1].strip()
    value = value.splitlines()[0].strip()
    value = re.sub(r"\s+", " ", value)
    return value[:80] if value else fallback


def build_llm_pipeline(model_name: str, max_new_tokens: int, temperature: float) -> Any:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
    except ImportError as exc:
        raise RuntimeError(
            "LLM labeling requires transformers, torch, and optional bitsandbytes support."
        ) from exc

    quantization_config = None
    if torch.cuda.is_available():
        try:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=getattr(torch, "bfloat16", torch.float16),
            )
        except Exception:
            quantization_config = None

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_kwargs = {"device_map": "auto"}
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    return pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        repetition_penalty=1.1,
    )


def build_ollama_labeler(
    model_name: str,
    host: str,
    temperature: float,
    max_new_tokens: int,
) -> Any:
    health = requests.get(f"{host.rstrip('/')}/api/tags", timeout=10)
    health.raise_for_status()

    def labeler(prompt: str) -> list[dict[str, str]]:
        response = requests.post(
            f"{host.rstrip('/')}/api/generate",
            json={
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_new_tokens,
                },
            },
            timeout=120,
        )
        response.raise_for_status()
        payload = response.json()
        return [{"generated_text": payload.get("response", "")}]

    return labeler


def generate_llm_label(
    topic_id: int,
    keywords: list[str],
    representative_docs: list[str],
    fallback_label: str,
    labeler: Any,
) -> str:
    doc_block = "\n".join(
        f"{idx}. {doc[:280].replace(chr(10), ' ')}"
        for idx, doc in enumerate(representative_docs[:3], start=1)
    )
    keyword_block = ", ".join(keywords[:8])
    prompt = (
        "You label social listening topics.\n"
        "Return only a short factual label with 2 to 6 words.\n\n"
        f"Keywords: {keyword_block}\n"
        f"Representative posts:\n{doc_block}\n\n"
        "Label:"
    )
    result = labeler(prompt)[0]["generated_text"]
    if result.startswith(prompt):
        result = result[len(prompt):]
    return clean_generated_label(result, fallback_label)


def summarize_topics(topic_model: Any, docs_df: pd.DataFrame, topics: list[int]) -> list[dict[str, Any]]:
    topic_counts = docs_df["topic_id"].value_counts().to_dict()
    summaries: list[dict[str, Any]] = []

    for topic_id in sorted(set(topics)):
        if topic_id == -1:
            summaries.append(
                {
                    "topic_id": -1,
                    "document_count": int(topic_counts.get(-1, 0)),
                    "keywords": [],
                    "representative_docs": docs_df.loc[
                        docs_df["topic_id"] == -1, "raw_text"
                    ].head(3).tolist(),
                    "default_name": "Outliers / Noise",
                }
            )
            continue

        keyword_pairs = topic_model.get_topic(topic_id) or []
        keywords = [word for word, _ in keyword_pairs[:10]]
        representative_docs = (topic_model.get_representative_docs(topic_id) or [])[:5]
        summaries.append(
            {
                "topic_id": int(topic_id),
                "document_count": int(topic_counts.get(topic_id, 0)),
                "keywords": keywords,
                "representative_docs": representative_docs,
                "default_name": build_keyword_label(keywords, topic_id),
            }
        )

    return summaries


def assign_topic_labels(
    topic_summaries: list[dict[str, Any]],
    label_mode: str,
    llm_provider: str,
    llm_model: str,
    label_cache_path: Path,
    llm_max_new_tokens: int,
    llm_temperature: float,
    ollama_host: str,
) -> tuple[dict[int, str], str]:
    label_map: dict[int, str] = {}

    if label_mode == "keywords":
        for summary in topic_summaries:
            label_map[summary["topic_id"]] = summary["default_name"]
        return label_map, "keywords"

    cache = load_label_cache(label_cache_path)
    cache["model_name"] = llm_model
    labeler = None

    try:
        if llm_provider == "ollama":
            labeler = build_ollama_labeler(
                model_name=llm_model,
                host=ollama_host,
                temperature=llm_temperature,
                max_new_tokens=llm_max_new_tokens,
            )
        else:
            labeler = build_llm_pipeline(
                model_name=llm_model,
                max_new_tokens=llm_max_new_tokens,
                temperature=llm_temperature,
            )
    except Exception as exc:
        print(f"LLM labeler unavailable, falling back to keyword labels: {exc}")
        for summary in topic_summaries:
            label_map[summary["topic_id"]] = summary["default_name"]
        return label_map, "keywords_fallback"

    for summary in topic_summaries:
        topic_id = summary["topic_id"]
        if topic_id == -1:
            label_map[topic_id] = "Outliers / Noise"
            continue

        signature = topic_signature(summary["keywords"], summary["representative_docs"])
        cached = cache["topics"].get(str(topic_id))
        if cached and cached.get("signature") == signature:
            label_map[topic_id] = cached["label"]
            continue

        fallback_label = summary["default_name"]
        try:
            label = generate_llm_label(
                topic_id=topic_id,
                keywords=summary["keywords"],
                representative_docs=summary["representative_docs"],
                fallback_label=fallback_label,
                labeler=labeler,
            )
        except Exception as exc:
            print(f"LLM labeling failed for topic {topic_id}, using keyword label instead: {exc}")
            label = fallback_label
        cache["topics"][str(topic_id)] = {"signature": signature, "label": label}
        label_map[topic_id] = label

    save_label_cache(label_cache_path, cache)
    return label_map, "llm"


def calculate_metrics(embeddings: np.ndarray, topic_ids: np.ndarray) -> dict[str, Any]:
    mask = topic_ids != -1
    clustered_embeddings = embeddings[mask]
    clustered_topics = topic_ids[mask]
    metrics: dict[str, Any] = {
        "total_documents": int(len(topic_ids)),
        "noise_documents": int((topic_ids == -1).sum()),
        "noise_ratio": round(float((topic_ids == -1).mean()), 4),
        "topic_count_excluding_noise": int(len(set(topic_ids)) - (1 if -1 in topic_ids else 0)),
    }

    if len(clustered_topics) > 1 and len(np.unique(clustered_topics)) > 1:
        sample_size = min(5000, len(clustered_topics))
        metrics["silhouette_score"] = round(
            float(
                silhouette_score(
                    clustered_embeddings,
                    clustered_topics,
                    sample_size=sample_size,
                    random_state=42,
                )
            ),
            4,
        )
        metrics["calinski_harabasz_score"] = round(
            float(calinski_harabasz_score(clustered_embeddings, clustered_topics)),
            2,
        )
    else:
        metrics["silhouette_score"] = None
        metrics["calinski_harabasz_score"] = None
    return metrics


def build_topic_model(
    BERTopic: Any,
    KeyBERTInspired: Any,
    MaximalMarginalRelevance: Any,
    hdbscan_module: Any,
    umap_module: Any,
    embedder: Any,
    min_cluster_size: int,
    min_samples: int,
    umap_n_neighbors: int,
    umap_components: int,
    umap_min_dist: float,
    min_df: int,
    max_df: float,
    nr_topics: int | str | None,
    top_n_words: int,
    doc_count: int | None = None,
) -> Any:
    safe_min_df: int | float = min_df
    safe_max_df: int | float = max_df
    if doc_count is not None and doc_count > 0:
        max_doc_freq = max_df * doc_count if isinstance(max_df, float) else max_df
        if max_doc_freq < min_df:
            safe_min_df = 1
            safe_max_df = 1.0

    vectorizer = CountVectorizer(
        stop_words="english",
        ngram_range=(1, 3),
        min_df=safe_min_df,
        max_df=safe_max_df,
    )
    umap_model = umap_module.UMAP(
        n_neighbors=umap_n_neighbors,
        n_components=umap_components,
        min_dist=umap_min_dist,
        metric="cosine",
        random_state=42,
    )
    hdbscan_model = hdbscan_module.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )
    representation_model = {
        "Main": KeyBERTInspired(),
        "MMR": MaximalMarginalRelevance(diversity=0.3),
    }
    return BERTopic(
        embedding_model=embedder,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer,
        representation_model=representation_model,
        verbose=True,
        nr_topics=nr_topics,
        top_n_words=top_n_words,
    )


def run_topic_model(
    docs_df: pd.DataFrame,
    docs: list[str],
    embeddings: np.ndarray,
    topic_model: Any,
    text_column: str,
) -> tuple[pd.DataFrame, list[int], Any]:
    topics, _ = topic_model.fit_transform(docs, embeddings)
    result_df = docs_df.copy()
    result_df["model_text"] = result_df[text_column]
    result_df["topic_id"] = topics
    return result_df, topics, topic_model


def calculate_dominance(topic_ids: list[int]) -> float:
    if not topic_ids:
        return 0.0
    counts = Counter(topic_ids)
    counts.pop(-1, None)
    if not counts:
        return 0.0
    return round(max(counts.values()) / len(topic_ids), 4)


def write_outputs(
    output_dir: Path,
    docs_df: pd.DataFrame,
    topic_summaries: list[dict[str, Any]],
    run_summary: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    docs_df.to_csv(output_dir / "bertopic_clustered_documents.csv", index=False)

    summary_df = pd.DataFrame(topic_summaries)
    summary_df["keywords"] = summary_df["keywords"].map(lambda items: ", ".join(items))
    summary_df["representative_docs"] = summary_df["representative_docs"].map(
        lambda items: "\n\n".join(items)
    )
    summary_df.to_csv(output_dir / "bertopic_topic_summary.csv", index=False)

    payload = {"summary": run_summary, "topics": topic_summaries}
    (output_dir / "bertopic_topic_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=True)
    )


def write_visualization(
    output_dir: Path,
    docs_df: pd.DataFrame,
    embeddings: np.ndarray,
    title: str,
    umap_module: Any,
    plotly_express: Any,
) -> None:
    reducer = umap_module.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
    )
    coords = reducer.fit_transform(embeddings)
    plot_df = docs_df.copy()
    plot_df["x"] = coords[:, 0]
    plot_df["y"] = coords[:, 1]

    fig = plotly_express.scatter(
        plot_df,
        x="x",
        y="y",
        color="topic_label",
        hover_data={
            "raw_text": True,
            "topic_id": True,
            "clean_text": False,
            "x": False,
            "y": False,
        },
        title=title,
        width=1200,
        height=800,
        template="plotly_white",
    )
    fig.update_traces(marker=dict(size=6, opacity=0.7))
    fig.write_html(output_dir / "bertopic_visualization.html")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BERTopic pipeline for social listening data.")
    parser.add_argument("--input", required=True, help="Path to JSON, CSV, or XLSX input.")
    parser.add_argument(
        "--input-type",
        choices=["auto", "json", "csv", "xlsx"],
        default="auto",
        help="Input format. Defaults to auto-detect from the file suffix.",
    )
    parser.add_argument("--column", default=None, help="Text column for CSV/XLSX inputs.")
    parser.add_argument(
        "--json-text-key",
        default="content",
        help="Text field to read from JSON post objects.",
    )
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/LaBSE",
        help="SentenceTransformer model used to create document embeddings.",
    )
    parser.add_argument(
        "--label-mode",
        choices=["keywords", "llm"],
        default="keywords",
        help="Use BERTopic keyword labels or optional local LLM labels.",
    )
    parser.add_argument(
        "--llm-provider",
        choices=["transformers", "ollama"],
        default="ollama",
        help="Backend used for --label-mode llm. Ollama is recommended on low-memory Macs.",
    )
    parser.add_argument(
        "--llm-model",
        default="qwen2.5:3b",
        help="Model id used when --label-mode llm. Example: qwen2.5:3b for Ollama.",
    )
    parser.add_argument(
        "--ollama-host",
        default="http://127.0.0.1:11434",
        help="Ollama server URL used when --llm-provider ollama.",
    )
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=5,
        help="Minimum cluster size passed to HDBSCAN.",
    )
    parser.add_argument(
        "--nr-topics",
        default=None,
        help="Optional BERTopic topic reduction target. Use an integer or 'auto'.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/bertopic_run",
        help="Directory for reports, CSVs, and visualization HTML.",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=2,
        help="Minimum samples passed to HDBSCAN.",
    )
    parser.add_argument(
        "--umap-n-neighbors",
        type=int,
        default=15,
        help="UMAP n_neighbors value for BERTopic.",
    )
    parser.add_argument(
        "--umap-components",
        type=int,
        default=5,
        help="UMAP n_components value for BERTopic fitting.",
    )
    parser.add_argument(
        "--umap-min-dist",
        type=float,
        default=0.0,
        help="UMAP min_dist value for BERTopic fitting.",
    )
    parser.add_argument(
        "--min-df",
        type=int,
        default=2,
        help="Minimum document frequency for c-TF-IDF vectorization.",
    )
    parser.add_argument(
        "--max-df",
        type=float,
        default=0.9,
        help="Maximum document frequency for c-TF-IDF vectorization.",
    )
    parser.add_argument(
        "--top-n-words",
        type=int,
        default=10,
        help="Number of topic keywords to retain.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Embedding batch size.",
    )
    parser.add_argument(
        "--no-dedupe",
        action="store_true",
        help="Keep exact duplicate raw texts instead of dropping them.",
    )
    parser.add_argument(
        "--no-visualize",
        action="store_true",
        help="Skip Plotly HTML visualization output.",
    )
    parser.add_argument(
        "--llm-max-new-tokens",
        type=int,
        default=24,
        help="Maximum label tokens generated in LLM mode.",
    )
    parser.add_argument(
        "--llm-temperature",
        type=float,
        default=0.05,
        help="Generation temperature for LLM topic labels.",
    )
    parser.add_argument(
        "--label-cache",
        default=None,
        help="Optional JSON cache path for LLM-generated topic labels.",
    )
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    BERTopic, KeyBERTInspired, MaximalMarginalRelevance, hdbscan, px, SentenceTransformer, umap = (
        ensure_base_dependencies()
    )

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    label_cache_path = (
        Path(args.label_cache) if args.label_cache else output_dir / "llm_label_cache.json"
    )

    docs_df = load_input(
        path=input_path,
        input_type=args.input_type,
        column=args.column,
        json_text_key=args.json_text_key,
    )
    docs_df = dedupe_documents(docs_df, enabled=not args.no_dedupe)
    docs = docs_df["clean_text"].tolist()

    print(f"Loaded {len(docs_df)} cleaned documents from {input_path.name}")
    print(f"Embedding model: {args.embedding_model}")

    embedder = load_embedder(SentenceTransformer, args.embedding_model)
    embeddings = embedder.encode(
        docs,
        show_progress_bar=True,
        batch_size=args.batch_size,
        normalize_embeddings=True,
    )

    vectorizer = CountVectorizer(
        stop_words="english",
        ngram_range=(1, 3),
        min_df=args.min_df,
        max_df=args.max_df,
    )
    umap_model = umap.UMAP(
        n_neighbors=args.umap_n_neighbors,
        n_components=args.umap_components,
        min_dist=args.umap_min_dist,
        metric="cosine",
        random_state=42,
    )
    hdbscan_model = hdbscan.HDBSCAN(
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )
    representation_model = {
        "Main": KeyBERTInspired(),
        "MMR": MaximalMarginalRelevance(diversity=0.3),
    }

    topic_model = BERTopic(
        embedding_model=embedder,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer,
        representation_model=representation_model,
        verbose=True,
        nr_topics=parse_nr_topics(args.nr_topics),
        top_n_words=args.top_n_words,
    )

    topics, _ = topic_model.fit_transform(docs, embeddings)
    docs_df["topic_id"] = topics

    topic_summaries = summarize_topics(topic_model, docs_df, topics)
    label_map, actual_label_mode = assign_topic_labels(
        topic_summaries=topic_summaries,
        label_mode=args.label_mode,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        label_cache_path=label_cache_path,
        llm_max_new_tokens=args.llm_max_new_tokens,
        llm_temperature=args.llm_temperature,
        ollama_host=args.ollama_host,
    )
    docs_df["topic_label"] = docs_df["topic_id"].map(label_map)
    docs_df["is_noise"] = docs_df["topic_id"] == -1

    for summary in topic_summaries:
        summary["topic_label"] = label_map[summary["topic_id"]]

    metrics = calculate_metrics(np.asarray(embeddings), np.asarray(topics))
    run_summary = {
        "input_path": str(input_path),
        "documents_used": int(len(docs_df)),
        "embedding_model": args.embedding_model,
        "label_mode_requested": args.label_mode,
        "label_mode_actual": actual_label_mode,
        "llm_model": args.llm_model if args.label_mode == "llm" else None,
        "llm_provider": args.llm_provider if args.label_mode == "llm" else None,
        "min_cluster_size": args.min_cluster_size,
        "nr_topics": parse_nr_topics(args.nr_topics),
        "output_dir": str(output_dir),
        **metrics,
    }

    write_outputs(output_dir=output_dir, docs_df=docs_df, topic_summaries=topic_summaries, run_summary=run_summary)
    if not args.no_visualize:
        write_visualization(
            output_dir=output_dir,
            docs_df=docs_df,
            embeddings=np.asarray(embeddings),
            title="BERTopic Social Listening Clusters",
            umap_module=umap,
            plotly_express=px,
        )

    print("\nBERTopic run complete")
    print("=" * 40)
    print(f"Documents used: {run_summary['documents_used']}")
    print(f"Topics found: {run_summary['topic_count_excluding_noise']}")
    print(f"Noise documents: {run_summary['noise_documents']}")
    print(f"Label mode used: {run_summary['label_mode_actual']}")
    print(f"Outputs written to: {output_dir}")

    preview = sorted(
        [summary for summary in topic_summaries if summary["topic_id"] != -1],
        key=lambda item: item["document_count"],
        reverse=True,
    )[:10]
    for summary in preview:
        print(
            f"- Topic {summary['topic_id']}: {summary['topic_label']} "
            f"({summary['document_count']} docs)"
        )


if __name__ == "__main__":
    main()
