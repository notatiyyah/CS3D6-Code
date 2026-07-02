import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Set, Tuple, Dict

from common.paths import PROCESSED, METRICS
from common.logging import setup_logger
from common.json_helpers import load_json, save_json
from eval.evaluators import RelationEvaluator, SpanEvaluator

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
    gs, ge = gold_span
    return any(max(0, min(ge, p["end"]) - max(gs, p["start"])) > 0 for p in pred_spans)


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

    pred_by_id: Dict[str, dict] = {d["id"]: d for d in pred_docs}

    # E2E relation metrics via RelationEvaluator
    rel_evaluator = RelationEvaluator(config.logger)
    e2e_results = rel_evaluator.evaluate(
        gold_docs,
        lambda gold_doc: get_predicted_relations(pred_by_id.get(gold_doc["id"], {})),
        gold_fn=get_gold_relations,
    )
    rel_evaluator.print_report(e2e_results, title="END-TO-END PIPELINE EVALUATION (RELATIONS)")

    # Span metrics via SpanEvaluator
    all_labels = set()
    for doc in gold_docs:
        all_labels.update(n["label"] for n in doc.get("needs", []))
        all_labels.update(p["label"] for p in doc.get("persons", []))

    span_evaluator = SpanEvaluator(all_labels, config.logger)
    gold_span_docs = [
        [(s["start"], s["end"], s["label"]) for s in d.get("needs", []) + d.get("persons", [])]
        for d in gold_docs
    ]
    pred_span_docs = [
        [(s["start"], s["end"], s["label"]) for s in d.get("needs", []) + d.get("persons", [])]
        for d in pred_docs
    ]
    span_results = span_evaluator.evaluate(gold_span_docs, pred_span_docs)
    span_evaluator.print_report(span_results, title="SPAN RECALL")

    # Conditional relation recall: of gold relations where both spans were found,
    # how many did the relation model link correctly?
    cond_tp, cond_fn = 0, 0
    for gold_doc, pred_doc in zip(gold_docs, pred_docs):
        pred_rels = get_predicted_relations(pred_doc)
        needs_pred = pred_doc.get("needs", [])
        persons_pred = pred_doc.get("persons", [])
        for need_id, person_id in [
            (str(r["from"]), str(r["to"])) for r in gold_doc.get("relations", [])
        ]:
            need_span = next((n for n in gold_doc.get("needs", []) if str(n["id"]) == need_id), None)
            person_span = next((p for p in gold_doc.get("persons", []) if str(p["id"]) == person_id), None)
            if need_span is None or person_span is None:
                continue
            if span_was_found((need_span["start"], need_span["end"]), needs_pred) and \
               span_was_found((person_span["start"], person_span["end"]), persons_pred):
                gold_pair = ((need_span["start"], need_span["end"]), (person_span["start"], person_span["end"]))
                cond_tp += gold_pair in pred_rels
                cond_fn += gold_pair not in pred_rels

    cond_rel_recall = cond_tp / (cond_tp + cond_fn) if (cond_tp + cond_fn) > 0 else 0.0
    print(f"Relation recall | spans found:    {cond_rel_recall:.4f}  ({cond_tp}/{cond_tp + cond_fn} linked correctly)\n")

    results = {
        "predictions_path": str(config.pred_path),
        "e2e": e2e_results,
        "span_metrics": span_results,
        "relation_recall_given_spans_found": {"tp": cond_tp, "fn": cond_fn, "recall": cond_rel_recall},
    }


    save_json(path=config.eval_path, data=results, logger=config.logger)
    config.logger.info("E2E Metrics saved to %s", config.eval_path)


if __name__ == "__main__":
    main()
