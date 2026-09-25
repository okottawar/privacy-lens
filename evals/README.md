# PrivacyLens Evaluation

The evaluation layer is intentionally separate from production analysis.
Curate manually reviewed privacy-policy cases before reporting benchmark numbers.
Do not treat generated labels as ground truth.

## Case format

Each JSONL case should contain:

```json
{
  "case_id": "retention-001",
  "category": "Retention",
  "relevant_chunk_ids": ["chunk_17", "chunk_21"]
}
```

## Retrieval metrics

- Recall@K — how many relevant chunks were retrieved.
- Precision@K — how much of the retrieved set was relevant.
- Reciprocal Rank — how early the first relevant chunk appeared.

## Analysis metrics to add

- citation accuracy
- severity/category accuracy
- structured-output validity
- confidence calibration
- end-to-end latency

The repository currently provides metric functions and test coverage but does not publish
benchmark scores until a manually reviewed case set is added.
