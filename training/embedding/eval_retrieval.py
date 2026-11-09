"""Retrieval eval for the embedding model (track A).

Scores a model on a fixed EmbeddingExample test set: each query is
ranked against the corpus (all positives + negatives in the file) and
we report recall@k and nDCG@k. With --baseline-model the same metrics
are computed for the baseline model and printed side by side — the
fine-tuned model only proceeds when it beats the baseline (see
docs/fine-tuning-lifecycle.md).

Model loading is lazy: --help works without torch. With no
sentence-transformers installed, exits with a clear message.

Usage:
    python -m training.embedding.eval_retrieval --data <file> --model <candidate>
    python -m training.embedding.eval_retrieval --data <file> --model <c> --baseline-model <b>
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _dcg(relevances: list[int]) -> float:
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances))


def recall_at_k(ranked_ids: list[str], positive_id: str, k: int) -> float:
    return 1.0 if positive_id in ranked_ids[:k] else 0.0


def ndcg_at_k(ranked_ids: list[str], positive_id: str, k: int) -> float:
    relevances = [1 if doc_id == positive_id else 0 for doc_id in ranked_ids[:k]]
    ideal = [1] + [0] * (k - 1)
    denom = _dcg(ideal)
    return _dcg(relevances) / denom if denom else 0.0


def evaluate(model, examples, corpus: list[tuple[str, str]], k: int) -> dict:
    """Rank each query against the corpus; return mean recall@k / nDCG@k."""
    corpus_texts = [text for _, text in corpus]
    corpus_embs = model.encode(corpus_texts, normalize_embeddings=True)
    recalls: list[float] = []
    ndcgs: list[float] = []
    for ex in examples:
        query_emb = model.encode([ex.query], normalize_embeddings=True)[0]
        scored = sorted(
            range(len(corpus)),
            key=lambda i: _cosine(list(query_emb), list(corpus_embs[i])),
            reverse=True,
        )
        ranked_ids = [corpus[i][0] for i in scored]
        positive_id = f"pos:{ex.id}"
        recalls.append(recall_at_k(ranked_ids, positive_id, k))
        ndcgs.append(ndcg_at_k(ranked_ids, positive_id, k))
    n = max(len(examples), 1)
    return {
        f"recall@{k}": round(sum(recalls) / n, 4),
        f"ndcg@{k}": round(sum(ndcgs) / n, 4),
        "n_queries": len(examples),
    }


def _load_model(name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(name)


def _log_mlflow(metrics: dict, model_name: str, tracking_uri: str) -> None:
    try:
        import mlflow

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("agentic-runtime-finetune")
        with mlflow.start_run(run_name=f"retrieval-eval-{Path(model_name).name}"):
            mlflow.log_params({"track": "embedding", "model": model_name})
            mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
    except Exception as exc:  # noqa: BLE001 — observability must never fail eval
        print(f"WARNING: could not log to MLflow ({exc}).")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Retrieval eval (recall@k / nDCG@k).")
    parser.add_argument("--data", required=True, help="EmbeddingExample JSONL (test split used).")
    parser.add_argument("--model", required=True, help="Candidate model name or path.")
    parser.add_argument("--baseline-model", default=None, help="Baseline model for comparison.")
    parser.add_argument("--k", type=int, default=5, help="k for recall@k / nDCG@k.")
    parser.add_argument("--tracking-uri", default="http://localhost:5000")
    parser.add_argument("--no-mlflow", action="store_true", help="Skip MLflow logging.")
    args = parser.parse_args(argv)

    try:
        from sentence_transformers import SentenceTransformer  # noqa: F401
    except ModuleNotFoundError as exc:
        print(f"ERROR: missing training dependency: {exc.name}")
        print("Install the training deps first:")
        print("  pip install -r training/requirements-training.txt")
        return 2

    from training.common.schemas import EmbeddingExample, validate_jsonl

    examples, errors = validate_jsonl(args.data, EmbeddingExample)
    if errors:
        print(f"ERROR: {len(errors)} invalid line(s) in {args.data}; fix the dataset first.")
        return 2
    test_examples = [ex for ex in examples if ex.split == "test"]
    if not test_examples:
        test_examples = examples
        print("NOTE: no test-split examples; evaluating on all examples.")
    corpus: list[tuple[str, str]] = []
    for ex in examples:
        corpus.append((f"pos:{ex.id}", ex.positive))
        for j, neg in enumerate(ex.negatives):
            corpus.append((f"neg:{ex.id}:{j}", neg))

    results: dict[str, dict] = {}
    for label, name in [("candidate", args.model), ("baseline", args.baseline_model)]:
        if not name:
            continue
        print(f"Loading {label} model: {name}")
        model = _load_model(name)
        results[label] = evaluate(model, test_examples, corpus, args.k)

    print(json.dumps(results, indent=2, ensure_ascii=False))
    if "candidate" in results and "baseline" in results:
        c, b = results["candidate"], results["baseline"]
        better = (
            c[f"recall@{args.k}"] >= b[f"recall@{args.k}"]
            and c[f"ndcg@{args.k}"] >= b[f"ndcg@{args.k}"]
        )
        verdict = (
            "VERDICT: candidate beats baseline"
            if better
            else "VERDICT: candidate does NOT beat baseline"
        )
        print(verdict)
    if not args.no_mlflow and "candidate" in results:
        _log_mlflow(results["candidate"], args.model, args.tracking_uri)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
