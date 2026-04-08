import argparse
import json
from pathlib import Path

import pandas as pd

from social_listening_bertopic import (
    build_ollama_labeler,
    clean_generated_label,
    generate_llm_label,
    load_label_cache,
    save_label_cache,
    topic_signature,
)


def load_topics(summary_path: Path) -> tuple[dict, list[dict]]:
    payload = json.loads(summary_path.read_text())
    return payload["summary"], payload["topics"]


def update_topic_labels(
    topics: list[dict],
    llm_model: str,
    ollama_host: str,
    label_cache_path: Path,
    max_new_tokens: int,
    temperature: float,
) -> dict[int, str]:
    cache = load_label_cache(label_cache_path)
    cache["model_name"] = llm_model
    labeler = build_ollama_labeler(
        model_name=llm_model,
        host=ollama_host,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
    )

    label_map: dict[int, str] = {}
    for topic in topics:
        topic_id = int(topic["topic_id"])
        if topic_id == -1:
            label_map[topic_id] = "Outliers / Noise"
            topic["topic_label"] = label_map[topic_id]
            continue

        keywords = topic.get("keywords") or []
        representative_docs = topic.get("representative_docs") or []
        fallback = topic.get("topic_label") or topic.get("default_name") or f"Topic {topic_id}"
        signature = topic_signature(keywords, representative_docs)
        cached = cache["topics"].get(str(topic_id))

        if cached and cached.get("signature") == signature:
            label = cached["label"]
        else:
            label = generate_llm_label(
                topic_id=topic_id,
                keywords=keywords,
                representative_docs=representative_docs,
                fallback_label=fallback,
                labeler=labeler,
            )
            label = clean_generated_label(label, fallback)
            cache["topics"][str(topic_id)] = {"signature": signature, "label": label}

        label_map[topic_id] = label
        topic["topic_label"] = label

    save_label_cache(label_cache_path, cache)
    return label_map


def update_summary(summary_path: Path, summary: dict, topics: list[dict], llm_model: str) -> None:
    summary["label_mode_requested"] = "llm"
    summary["label_mode_actual"] = "llm"
    summary["llm_model"] = llm_model
    summary["llm_provider"] = "ollama"
    summary_path.write_text(json.dumps({"summary": summary, "topics": topics}, indent=2, ensure_ascii=True))


def update_clustered_documents(clustered_csv_path: Path, label_map: dict[int, str]) -> None:
    df = pd.read_csv(clustered_csv_path)
    df["topic_label"] = df["topic_id"].map(lambda value: label_map.get(int(value), str(value)))
    df.to_csv(clustered_csv_path, index=False)


def update_topic_summary_csv(topic_summary_csv_path: Path, topics: list[dict]) -> None:
    summary_df = pd.DataFrame(topics)
    summary_df["keywords"] = summary_df["keywords"].map(lambda items: ", ".join(items))
    summary_df["representative_docs"] = summary_df["representative_docs"].map(
        lambda items: "\n\n".join(items)
    )
    summary_df.to_csv(topic_summary_csv_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply local LLM labels to existing BERTopic outputs.")
    parser.add_argument("--output-dir", required=True, help="Directory containing BERTopic output files.")
    parser.add_argument("--llm-model", default="qwen2.5:3b", help="Ollama model to use for topic labels.")
    parser.add_argument(
        "--ollama-host",
        default="http://127.0.0.1:11434",
        help="Ollama server URL.",
    )
    parser.add_argument("--llm-max-new-tokens", type=int, default=24)
    parser.add_argument("--llm-temperature", type=float, default=0.05)
    parser.add_argument("--label-cache", default=None, help="Optional cache path.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    summary_path = output_dir / "bertopic_topic_summary.json"
    clustered_csv_path = output_dir / "bertopic_clustered_documents.csv"
    topic_summary_csv_path = output_dir / "bertopic_topic_summary.csv"
    label_cache_path = Path(args.label_cache) if args.label_cache else output_dir / "llm_label_cache.json"

    summary, topics = load_topics(summary_path)
    label_map = update_topic_labels(
        topics=topics,
        llm_model=args.llm_model,
        ollama_host=args.ollama_host,
        label_cache_path=label_cache_path,
        max_new_tokens=args.llm_max_new_tokens,
        temperature=args.llm_temperature,
    )
    update_summary(summary_path, summary, topics, args.llm_model)
    update_clustered_documents(clustered_csv_path, label_map)
    update_topic_summary_csv(topic_summary_csv_path, topics)
    print(f"Updated BERTopic outputs in {output_dir} with Ollama labels.")


if __name__ == "__main__":
    main()
