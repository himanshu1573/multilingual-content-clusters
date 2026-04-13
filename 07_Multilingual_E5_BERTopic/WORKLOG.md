# Worklog: Multilingual E5 BERTopic Track

Date: 2026-04-13

## Purpose

Create a fresh, non-destructive clustering path that switches the main semantic model from LaBSE to `intfloat/multilingual-e5-large`.

## Actions Completed

1. Created a new folder:
   `07_Multilingual_E5_BERTopic`
2. Copied the BERTopic baseline into a fresh script:
   `07_Multilingual_E5_BERTopic/multilingual_e5_bertopic.py`
3. Updated the new script so that:
   - default embedder is `intfloat/multilingual-e5-large`
   - E5 inputs are prefixed with `passage:`
   - the main execution path uses BERTopic outlier reduction
   - run summaries store key proof fields for future reference
4. Added this worklog and a README in the same folder to preserve context and proof.
5. Attempted the first `multilingual-e5-large` run and found that the copied loader was forcing offline-only model loading.
6. Patched the fresh script so it now:
   - tries local cached weights first
   - falls back to an online download when the model is not cached
7. Fixed a BERTopic post-processing bug after the first successful clustering pass:
   - `get_representative_docs(topic_id)` was being treated like a list
   - in this run shape, BERTopic returned representative docs through a mapping structure
   - summary generation now reads the full representative-doc map safely

## Technical Decisions

- Keep the earlier folders untouched.
- Stay semantic-first and treat NER as support, not the main clustering key.
- Keep BERTopic because it already fits the repository and supports multilingual short text reasonably well.
- Use embedding-based outlier reassignment because noise was a known weakness in prior runs.

## Proof Fields Captured In Future Runs

The generated `bertopic_topic_summary.json` from this track will include:

- `embedding_model`
- `embedding_input_prefix`
- `min_cluster_size`
- `min_samples`
- `nr_topics`
- `outlier_reduction_strategy`
- clustering metrics and noise counts

## Next Validation Step

Run the new pipeline on `00_Shared_Data/Video Titles.xlsx` and compare:

- noise ratio
- topic count
- silhouette score
- top-topic readability
- overlap versus previous LaBSE-based outputs

## Run Notes

- First execution attempt timestamp: `2026-04-13 00:20:12 IST`
- First failure reason: `multilingual-e5-large` was not cached locally and the inherited loader was still using offline-only behavior.
- Second failure reason: summary generation crashed on representative-doc access after outlier reduction and topic reduction.
- Current status: loader fixed and representative-doc handling fixed, ready for rerun.
