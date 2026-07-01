import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Set, Tuple, Dict

from common.paths import PROCESSED, METRICS
from common.logging import setup_logger
from common.json_helpers import load_json, save_json
from shared.span_model import spans_overlap

SpanPair = Tuple[Tuple[int, int], Tuple[int, int]]


@dataclass
class Config:
    pred_path: Path
    gt_path: Path = PROCESSED / "val_data.json"

    def __post_init__(self):
        self.pred_path = Path(self.pred_path)
        self.eval_path = METRICS / "e2e_metrics.json"
        self.logger = setup_logger("eval.e2e_metrics", "eval_e2e_metrics.log")


def get_gold_relations(gold_doc: dict) -> Set[SpanPair]:
    needs = {str(n["id"]): (n["start"], n["end"]) for n in gold_doc.get("needs", [])}
    persons = {str(p["id"]): (p["start"], p["end"]) for p in gold_doc.get("persons", [])}
    return {
        (needs[str(rel["from"])], persons[str(rel["to"])])
        for rel in gold_doc.get("relations", [])
        if str(rel["from"]) in needs and str(rel["to"]) in persons
    }


def get_predicted_relations(pred_doc: dict) -> Set[SpanPair]:
    pred_spans: Dict[str, Tuple[int, int]] = {
        item["id"]: (item["start"], item["end"])
        for item in pred_doc.get("needs", []) + pred_doc.get("persons", [])
    }
    return {
        (pred_spans[rel[0]], pred_spans[rel[1]])
        for rel in pred_doc.get("relations", [])
        if rel[0] in pred_spans and rel[1] in pred_spans
    }


def span_was_found(gold_span: Tuple[int, int], pred_spans: list) -> bool:
    """True if any predicted span loosely overlaps the gold span."""
    gs, ge = gold_span
    return any(spans_overlap(gs, ge, p["start"], p["end"]) for p in pred_spans)


def main():
    if len(sys.argv) < 2:
        print("Usage: python eval_e2e.py <path/to/predictions>")
        sys.exit(1)

    config = Config(pred_path=Path(sys.argv[1]))
    config.logger.info("Evaluating End-to-End Pipeline: %s", config.pred_path)

    gold_docs = load_json(config.gt_path, config.logger)
    pred_docs = load_json(config.pred_path, config.logger)

    if len(gold_docs) != len(pred_docs):
        config.logger.error("Mismatch: Gold (%d) != Pred (%d)", len(gold_docs), len(pred_docs))
        return

    # E2E relation metrics (strict span boundaries)
    total_tp, total_fp, total_fn = 0, 0, 0

    # Span recall
    gold_spans_total, gold_spans_found = 0, 0

    # Relation recall conditioned on both spans being found
    cond_tp, cond_fn = 0, 0

    for gold_doc, pred_doc in zip(gold_docs, pred_docs):
        gold_rels = get_gold_relations(gold_doc)
        pred_rels = get_predicted_relations(pred_doc)

        total_tp += len(gold_rels & pred_rels)
        total_fp += len(pred_rels - gold_rels)
        total_fn += len(gold_rels - pred_rels)

        # Span recall: did the span classifier find each gold need/person?
        all_pred_spans = pred_doc.get("needs", []) + pred_doc.get("persons", [])
        gold_spans = (
            [(n["start"], n["end"]) for n in gold_doc.get("needs", [])] +
            [(p["start"], p["end"]) for p in gold_doc.get("persons", [])]
        )
        gold_spans_total += len(gold_spans)
        gold_spans_found += sum(span_was_found(s, all_pred_spans) for s in gold_spans)

        # Conditional relation recall: of gold relations where both spans were found,
        # how many did the relation model link correctly?
        needs_pred = pred_doc.get("needs", [])
        persons_pred = pred_doc.get("persons", [])
        for need_id, person_id in [
            (str(r["from"]), str(r["to"])) for r in gold_doc.get("relations", [])
        ]:
            need_span = next((n for n in gold_doc.get("needs", []) if str(n["id"]) == need_id), None)
            person_span = next((p for p in gold_doc.get("persons", []) if str(p["id"]) == person_id), None)
            if need_span is None or person_span is None:
                continue
            need_found = span_was_found((need_span["start"], need_span["end"]), needs_pred)
            person_found = span_was_found((person_span["start"], person_span["end"]), persons_pred)
            if need_found and person_found:
                gold_pair = ((need_span["start"], need_span["end"]), (person_span["start"], person_span["end"]))
                if gold_pair in pred_rels:
                    cond_tp += 1
                else:
                    cond_fn += 1

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    span_recall = gold_spans_found / gold_spans_total if gold_spans_total > 0 else 0.0
    cond_rel_recall = cond_tp / (cond_tp + cond_fn) if (cond_tp + cond_fn) > 0 else 0.0

    results = {
        "predictions_path": str(config.pred_path),
        "e2e": {"precision": precision, "recall": recall, "f1": f1,
                "tp": total_tp, "fp": total_fp, "fn": total_fn},
        "span_recall": {"found": gold_spans_found, "total": gold_spans_total, "recall": span_recall},
        "relation_recall_given_spans_found": {"tp": cond_tp, "fn": cond_fn, "recall": cond_rel_recall},
    }

    print(f"\n=== END-TO-END PIPELINE EVALUATION ===")
    print(f"E2E  — Precision: {precision:.4f}  Recall: {recall:.4f}  F1: {f1:.4f}")
    print(f"       TP: {total_tp}  FP: {total_fp}  FN: {total_fn}")
    print(f"\nSpan recall (loose):              {span_recall:.4f}  ({gold_spans_found}/{gold_spans_total} gold spans found)")
    print(f"Relation recall | spans found:    {cond_rel_recall:.4f}  ({cond_tp}/{cond_tp + cond_fn} linked correctly)\n")

    save_json(path=config.eval_path, data=results, logger=config.logger)
    config.logger.info("E2E Metrics saved to %s", config.eval_path)


if __name__ == "__main__":
    main()
