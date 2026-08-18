"""Evaluation harness for the ontology deep-search engine.

Samples answerable questions across types plus a batch of unanswerable ones, runs
the query engine over the live graph, and scores real numbers:

- person / pr / url / company: set precision / recall / F1 against ground truth,
  with type-aware normalization (eids exact; links trailing-slash-insensitive;
  companies case-insensitive).
- content: an LLM judge makes a correct / incorrect call against the reference.
- unanswerable: correct iff the engine abstained.

Results (per-type F1, content accuracy, unanswerable abstention rate, macro
summary, and every per-question row) are written to data/eval_results.json.
Throughput comes from a thread pool; each answer is an independent HTTP + LLM
round-trip, so it parallelizes cleanly.
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import llm, query
from .graph import OntologyGraph

_DATA = Path(__file__).resolve().parents[2] / "data"
_SET_TYPES = ("person", "pr", "url", "company")


def _norm(qtype: str, v: str) -> str:
    s = str(v).strip()
    if qtype in ("url", "pr"):
        return s.rstrip("/").lower()
    if qtype == "company":
        return s.lower()
    return s  # eids: exact


def _prf1(pred, gt, qtype):
    p_set = {_norm(qtype, x) for x in pred}
    g_set = {_norm(qtype, x) for x in gt}
    if not g_set:
        return None
    inter = p_set & g_set
    prec = len(inter) / len(p_set) if p_set else 0.0
    rec = len(inter) / len(g_set)
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"precision": prec, "recall": rec, "f1": f1,
            "n_pred": len(p_set), "n_gt": len(g_set), "n_correct": len(inter)}


def _judge_content(question: str, predicted: str, reference: str, model=None) -> bool:
    if not predicted:
        return False
    prompt = (
        "You grade an answer against a reference. The answer is CORRECT if it "
        "captures the substance of the reference (the key facts / features), even "
        "if worded differently or less complete. It is INCORRECT if it contradicts "
        "the reference or misses its main content.\n\n"
        f"Question: {question}\n\nReference answer:\n{reference[:1800]}\n\n"
        f"Model answer:\n{predicted[:1800]}\n\n"
        'Return ONLY JSON {"correct": true|false}.'
    )
    try:
        r = llm.ask_json(prompt, system="You are a strict grader. Output only JSON.", model=model)
        return bool(r.get("correct")) if isinstance(r, dict) else False
    except Exception:
        return False


def _sample(questions, limit_per_type, include_unanswerable, seed):
    rng = random.Random(seed)
    by_type = defaultdict(list)
    for q in questions:
        if q["answerable"] and q["type"]:
            by_type[q["type"]].append(q)
    picked = []
    for t, qs in by_type.items():
        rng.shuffle(qs)
        picked.extend(qs[:limit_per_type])
    unans = [q for q in questions if not q["answerable"]]
    rng.shuffle(unans)
    picked.extend(unans[:include_unanswerable])
    return picked


def _score(q, result, model):
    qtype = q["type"]
    row = {"product": q["product"], "type": qtype, "question": q["question"],
           "abstain": result["abstain"]}
    if not q["answerable"]:
        row["correct"] = bool(result["abstain"])
        return row
    if qtype == "content":
        pred = result["answer"] if isinstance(result["answer"], str) else ""
        row["correct"] = _judge_content(q["question"], pred, "\n".join(q["ground_truth"])
                                         if isinstance(q["ground_truth"], list) else str(q["ground_truth"]),
                                         model=model)
        row["predicted"] = pred[:300]
        return row
    pred = result["answer"] if isinstance(result["answer"], list) else []
    m = _prf1(pred, q["ground_truth"], qtype)
    row.update(m or {})
    return row


def run(limit_per_type: int = 20, include_unanswerable: int = 40, model=None,
        workers: int = 16, seed: int = 0, out_path: str | None = None) -> dict:
    with open(_DATA / "ontology.json", encoding="utf-8") as fh:
        onto = json.load(fh)
    sample = _sample(onto["questions"], limit_per_type, include_unanswerable, seed)
    graph = OntologyGraph()

    def work(q):
        try:
            res = query.answer(q["question"], graph, model=model)
        except Exception as e:  # a single failure must not sink the run
            res = {"type": q["type"], "product": q["product"], "answer": [],
                   "abstain": True, "error": str(e)[:200]}
        return _score(q, res, model)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        rows = list(ex.map(work, sample))

    report = _aggregate(rows)
    report["config"] = {"limit_per_type": limit_per_type,
                        "include_unanswerable": include_unanswerable, "seed": seed}
    out = Path(out_path) if out_path else (_DATA / "eval_results.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"summary": report, "rows": rows}, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    return report


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def _aggregate(rows) -> dict:
    per_type = {}
    for t in _SET_TYPES:
        rs = [r for r in rows if r["type"] == t and r.get("f1") is not None]
        if rs:
            per_type[t] = {
                "n": len(rs),
                "f1": round(_mean([r["f1"] for r in rs]), 4),
                "precision": round(_mean([r["precision"] for r in rs]), 4),
                "recall": round(_mean([r["recall"] for r in rs]), 4),
                "abstain_rate": round(_mean([1.0 if r["abstain"] else 0.0 for r in rs]), 4),
            }
    content = [r for r in rows if r["type"] == "content" and "correct" in r]
    unans = [r for r in rows if r.get("correct") is not None and r["type"] is None]
    summary = {"per_type": per_type}
    if content:
        summary["content"] = {"n": len(content),
                              "accuracy": round(_mean([1.0 if r["correct"] else 0.0 for r in content]), 4)}
    if unans:
        summary["unanswerable"] = {"n": len(unans),
                                   "abstention_rate": round(_mean([1.0 if r["correct"] else 0.0 for r in unans]), 4)}
    f1s = [v["f1"] for v in per_type.values()]
    summary["macro_f1_set_types"] = round(_mean(f1s), 4) if f1s else 0.0
    return summary


if __name__ == "__main__":
    import sys
    lpt = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    una = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    rep = run(limit_per_type=lpt, include_unanswerable=una)
    print(json.dumps(rep, indent=2))

