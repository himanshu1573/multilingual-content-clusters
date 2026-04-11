import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from social_listening_bertopic import (
    assign_topic_labels,
    build_topic_model,
    calculate_dominance,
    calculate_metrics,
    dedupe_documents,
    ensure_base_dependencies,
    load_input,
    load_embedder,
    parse_nr_topics,
    run_topic_model,
    summarize_topics,
    write_outputs,
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


def run_single_experiment(
    experiment_name: str,
    docs_df: pd.DataFrame,
    text_column: str,
    embeddings: np.ndarray,
    embedder: Any,
    deps: tuple[Any, Any, Any, Any, Any, Any, Any],
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, Any]:
    BERTopic, KeyBERTInspired, MaximalMarginalRelevance, hdbscan_module, px, _, umap_module = deps
    docs = docs_df[text_column].tolist()
    topic_model = build_topic_model(
        BERTopic=BERTopic,
        KeyBERTInspired=KeyBERTInspired,
        MaximalMarginalRelevance=MaximalMarginalRelevance,
        hdbscan_module=hdbscan_module,
        umap_module=umap_module,
        embedder=embedder,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        umap_n_neighbors=args.umap_n_neighbors,
        umap_components=args.umap_components,
        umap_min_dist=args.umap_min_dist,
        min_df=args.min_df,
        max_df=args.max_df,
        nr_topics=parse_nr_topics(args.nr_topics),
        top_n_words=args.top_n_words,
        doc_count=len(docs_df),
    )
    result_df, topics, fitted_model = run_topic_model(
        docs_df=docs_df,
        docs=docs,
        embeddings=embeddings,
        topic_model=topic_model,
        text_column=text_column,
    )
    topic_summaries = summarize_topics(fitted_model, result_df, topics)
    label_map, actual_label_mode = assign_topic_labels(
        topic_summaries=topic_summaries,
        label_mode=args.label_mode,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        label_cache_path=output_dir / "llm_label_cache.json",
        llm_max_new_tokens=args.llm_max_new_tokens,
        llm_temperature=args.llm_temperature,
        ollama_host=args.ollama_host,
    )
    result_df["topic_label"] = result_df["topic_id"].map(label_map)
    result_df["is_noise"] = result_df["topic_id"] == -1
    for summary in topic_summaries:
        summary["topic_label"] = label_map[summary["topic_id"]]

    metrics = calculate_metrics(embeddings, np.asarray(topics))
    run_summary = {
        "experiment_name": experiment_name,
        "text_column": text_column,
        "documents_used": int(len(result_df)),
        "embedding_model": args.embedding_model,
        "label_mode_requested": args.label_mode,
        "label_mode_actual": actual_label_mode,
        "llm_model": args.llm_model if args.label_mode == "llm" else None,
        "llm_provider": args.llm_provider if args.label_mode == "llm" else None,
        "min_cluster_size": args.min_cluster_size,
        "min_samples": args.min_samples,
        "nr_topics": parse_nr_topics(args.nr_topics),
        "dominance_ratio": calculate_dominance(topics),
        "output_dir": str(output_dir),
        **metrics,
    }
    write_outputs(output_dir=output_dir, docs_df=result_df, topic_summaries=topic_summaries, run_summary=run_summary)
    return {
        "summary": run_summary,
        "top_topics": [
            {
                "topic_id": topic["topic_id"],
                "topic_label": topic["topic_label"],
                "document_count": topic["document_count"],
            }
            for topic in sorted(topic_summaries, key=lambda item: item["document_count"], reverse=True)[:10]
        ],
    }


def run_hybrid_experiment(
    experiment_name: str,
    docs_df: pd.DataFrame,
    text_column: str,
    embeddings: np.ndarray,
    embedder: Any,
    deps: tuple[Any, Any, Any, Any, Any, Any, Any],
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, Any]:
    BERTopic, KeyBERTInspired, MaximalMarginalRelevance, hdbscan_module, px, _, umap_module = deps
    result_df = docs_df.copy()
    result_df["model_text"] = result_df[text_column]
    result_df["topic_id"] = -1
    result_df["topic_label"] = "Outliers / Noise"
    result_df["is_noise"] = True

    topic_summaries: list[dict[str, Any]] = []
    next_topic_id = 0

    for bucket_name, bucket_df in docs_df.groupby("coarse_bucket"):
        if len(bucket_df) < args.min_cluster_size:
            continue

        bucket_indices = bucket_df.index.to_list()
        bucket_embeddings = embeddings[bucket_indices]
        bucket_docs = bucket_df[text_column].tolist()
        bucket_model = build_topic_model(
            BERTopic=BERTopic,
            KeyBERTInspired=KeyBERTInspired,
            MaximalMarginalRelevance=MaximalMarginalRelevance,
            hdbscan_module=hdbscan_module,
            umap_module=umap_module,
            embedder=embedder,
            min_cluster_size=args.min_cluster_size,
            min_samples=args.min_samples,
            umap_n_neighbors=args.umap_n_neighbors,
            umap_components=args.umap_components,
            umap_min_dist=args.umap_min_dist,
            min_df=args.min_df,
            max_df=args.max_df,
            nr_topics=parse_nr_topics(args.nr_topics),
            top_n_words=args.top_n_words,
            doc_count=len(bucket_df),
        )
        try:
            bucket_result, bucket_topics, fitted_model = run_topic_model(
                docs_df=bucket_df,
                docs=bucket_docs,
                embeddings=bucket_embeddings,
                topic_model=bucket_model,
                text_column=text_column,
            )
        except ValueError as exc:
            if "max_df corresponds to < documents than min_df" not in str(exc):
                raise
            bucket_model = build_topic_model(
                BERTopic=BERTopic,
                KeyBERTInspired=KeyBERTInspired,
                MaximalMarginalRelevance=MaximalMarginalRelevance,
                hdbscan_module=hdbscan_module,
                umap_module=umap_module,
                embedder=embedder,
                min_cluster_size=args.min_cluster_size,
                min_samples=args.min_samples,
                umap_n_neighbors=args.umap_n_neighbors,
                umap_components=args.umap_components,
                umap_min_dist=args.umap_min_dist,
                min_df=1,
                max_df=1.0,
                nr_topics=parse_nr_topics(args.nr_topics),
                top_n_words=args.top_n_words,
                doc_count=len(bucket_df),
            )
            bucket_result, bucket_topics, fitted_model = run_topic_model(
                docs_df=bucket_df,
                docs=bucket_docs,
                embeddings=bucket_embeddings,
                topic_model=bucket_model,
                text_column=text_column,
            )
        bucket_summaries = summarize_topics(fitted_model, bucket_result, bucket_topics)
        local_to_global: dict[int, int] = {}

        for summary in bucket_summaries:
            local_topic_id = summary["topic_id"]
            if local_topic_id == -1:
                continue
            global_topic_id = next_topic_id
            next_topic_id += 1
            local_to_global[local_topic_id] = global_topic_id
            summary["topic_id"] = global_topic_id
            summary["bucket_name"] = bucket_name
            summary["default_name"] = f"{bucket_name} / {summary['default_name']}"
            topic_summaries.append(summary)

        for index, local_topic_id in zip(bucket_result.index, bucket_topics):
            if local_topic_id == -1:
                continue
            global_topic_id = local_to_global[local_topic_id]
            result_df.loc[index, "topic_id"] = global_topic_id

    final_topics = result_df["topic_id"].tolist()
    label_map, actual_label_mode = assign_topic_labels(
        topic_summaries=topic_summaries,
        label_mode=args.label_mode,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        label_cache_path=output_dir / "llm_label_cache.json",
        llm_max_new_tokens=args.llm_max_new_tokens,
        llm_temperature=args.llm_temperature,
        ollama_host=args.ollama_host,
    )
    label_map[-1] = "Outliers / Noise"
    result_df["topic_label"] = result_df["topic_id"].map(label_map)
    result_df["is_noise"] = result_df["topic_id"] == -1
    for summary in topic_summaries:
        summary["topic_label"] = label_map[summary["topic_id"]]

    noise_docs = result_df.loc[result_df["topic_id"] == -1, "raw_text"].head(3).tolist()
    topic_summaries.insert(
        0,
        {
            "topic_id": -1,
            "document_count": int((result_df["topic_id"] == -1).sum()),
            "keywords": [],
            "representative_docs": noise_docs,
            "default_name": "Outliers / Noise",
            "topic_label": "Outliers / Noise",
            "bucket_name": "noise",
        },
    )

    metrics = calculate_metrics(embeddings, np.asarray(final_topics))
    run_summary = {
        "experiment_name": experiment_name,
        "text_column": text_column,
        "documents_used": int(len(result_df)),
        "embedding_model": args.embedding_model,
        "label_mode_requested": args.label_mode,
        "label_mode_actual": actual_label_mode,
        "llm_model": args.llm_model if args.label_mode == "llm" else None,
        "llm_provider": args.llm_provider if args.label_mode == "llm" else None,
        "min_cluster_size": args.min_cluster_size,
        "min_samples": args.min_samples,
        "nr_topics": parse_nr_topics(args.nr_topics),
        "dominance_ratio": calculate_dominance(final_topics),
        "bucket_count": int(result_df["coarse_bucket"].nunique()),
        "output_dir": str(output_dir),
        **metrics,
    }
    write_outputs(output_dir=output_dir, docs_df=result_df, topic_summaries=topic_summaries, run_summary=run_summary)
    return {
        "summary": run_summary,
        "top_topics": [
            {
                "topic_id": topic["topic_id"],
                "topic_label": topic["topic_label"],
                "document_count": topic["document_count"],
            }
            for topic in sorted(topic_summaries, key=lambda item: item["document_count"], reverse=True)[:10]
        ],
    }


def write_comparison(output_root: Path, experiment_results: list[dict[str, Any]]) -> None:
    rows = []
    for result in experiment_results:
        summary = result["summary"]
        rows.append(
            {
                "experiment_name": summary["experiment_name"],
                "documents_used": summary["documents_used"],
                "topic_count_excluding_noise": summary["topic_count_excluding_noise"],
                "noise_ratio": summary["noise_ratio"],
                "silhouette_score": summary["silhouette_score"],
                "calinski_harabasz_score": summary["calinski_harabasz_score"],
                "dominance_ratio": summary["dominance_ratio"],
                "output_dir": summary["output_dir"],
            }
        )
    pd.DataFrame(rows).to_csv(output_root / "comparison_summary.csv", index=False)
    (output_root / "comparison_summary.json").write_text(
        json.dumps({"experiments": experiment_results}, indent=2, ensure_ascii=True)
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare BERTopic strategy variants on the same dataset.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--input-type", choices=["auto", "json", "csv", "xlsx"], default="auto")
    parser.add_argument("--column", default=None)
    parser.add_argument("--json-text-key", default="content")
    parser.add_argument("--embedding-model", default="sentence-transformers/LaBSE")
    parser.add_argument("--output-root", default="outputs/strategy_compare")
    parser.add_argument("--label-mode", choices=["keywords", "llm"], default="keywords")
    parser.add_argument("--llm-provider", choices=["transformers", "ollama"], default="ollama")
    parser.add_argument("--llm-model", default="qwen2.5:3b")
    parser.add_argument("--ollama-host", default="http://127.0.0.1:11434")
    parser.add_argument("--min-cluster-size", type=int, default=25)
    parser.add_argument("--min-samples", type=int, default=8)
    parser.add_argument("--nr-topics", default="none")
    parser.add_argument("--umap-n-neighbors", type=int, default=15)
    parser.add_argument("--umap-components", type=int, default=5)
    parser.add_argument("--umap-min-dist", type=float, default=0.0)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--max-df", type=float, default=0.9)
    parser.add_argument("--top-n-words", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--llm-max-new-tokens", type=int, default=24)
    parser.add_argument("--llm-temperature", type=float, default=0.05)
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    deps = ensure_base_dependencies()
    _, _, _, _, _, SentenceTransformer, _ = deps

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    docs_df = load_input(
        path=Path(args.input),
        input_type=args.input_type,
        column=args.column,
        json_text_key=args.json_text_key,
    )
    docs_df = dedupe_documents(docs_df, enabled=True)

    embedder = load_embedder(SentenceTransformer, args.embedding_model)
    clean_embeddings = prepare_embeddings(embedder, docs_df["clean_text"].tolist(), args.batch_size)
    ner_embeddings = prepare_embeddings(embedder, docs_df["ner_text"].tolist(), args.batch_size)

    results = []
    print("Running Option A: tuned BERTopic")
    results.append(
        run_single_experiment(
            experiment_name="option_a_tuned",
            docs_df=docs_df,
            text_column="clean_text",
            embeddings=clean_embeddings,
            embedder=embedder,
            deps=deps,
            args=args,
            output_dir=output_root / "option_a_tuned",
        )
    )

    print("Running Option B: NER-normalized BERTopic")
    results.append(
        run_single_experiment(
            experiment_name="option_b_ner",
            docs_df=docs_df,
            text_column="ner_text",
            embeddings=ner_embeddings,
            embedder=embedder,
            deps=deps,
            args=args,
            output_dir=output_root / "option_b_ner",
        )
    )

    print("Running Option C: hybrid bucketed BERTopic")
    results.append(
        run_hybrid_experiment(
            experiment_name="option_c_hybrid",
            docs_df=docs_df,
            text_column="ner_text",
            embeddings=ner_embeddings,
            embedder=embedder,
            deps=deps,
            args=args,
            output_dir=output_root / "option_c_hybrid",
        )
    )

    write_comparison(output_root, results)
    print(f"Wrote comparison outputs to {output_root}")


if __name__ == "__main__":
    main()
