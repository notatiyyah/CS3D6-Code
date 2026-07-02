import numpy as np
from tabulate import tabulate
from collections import defaultdict
from sklearn.metrics import f1_score, precision_score, recall_score

class SpanEvaluator():
    """
    Evaluator for span-level NER extraction (Loose, Strict, and IoU-thresholded
    matching).

    Loose: any character overlap + same label counts as a match.
    Strict: exact boundary match (start, end) + same label.
    IoU: Configurable (see self.iou_thresholds) 'overlap ratio' fraction.

    All three modes use the same greedy best-match-first assignment: all
    candidate (gold, pred) pairs are collected, sorted by match quality
    (IoU score for IoU mode; binary 1/0 for loose/strict), then assigned
    greedily so each gold span and each prediction is used at most once.
    """

    # IoU thresholds to sweep. 0.0 is equivalent to loose and 1.0 is equivalent to strict.
    iou_thresholds = (0.3, 0.5, 0.7, 0.9)

    def __init__(self, all_labels, logger):
        self.all_labels = sorted(list(all_labels))
        self.logger = logger

    def _compute_metrics(self, tp, fp, fn):
        p = tp / (tp + fp) if tp + fp else 0
        r = tp / (tp + fn) if tp + fn else 0
        f1 = 2 * p * r / (p + r) if p + r else 0

        return p, r, f1

    def _match_and_update(self, tracker, true_spans, pred_spans, score_fn, threshold=0.0):
        """
        Greedy matcher. Computes scores for all valid (same-label) pairs, sorts descending by score, and assigns matches greedily.
        Needed for models that output overlapping spans.

        Example: gt: "foster care" and pred -> "foster" and "care" and "foster care"
        We want to give the "foster care" pred (which is correct) the right score vs the two preds "foster" and "care" (which are incorrect).
        If it was random or chronological, we may penalise the correct span.
        """
        candidates = []
        for pi, (p_start, p_end, p_label) in enumerate(pred_spans):
            for ti, (t_start, t_end, t_label) in enumerate(true_spans):
                if p_label != t_label:
                    continue
                score = score_fn(p_start, p_end, t_start, t_end)
                # Must have a score strictly > 0 (meaning overlap exists) and meet the threshold
                if score > 0 and score >= threshold:
                    candidates.append((score, ti, pi, p_label))

        # Sort by score descending for greedy best-match-first assignment
        candidates.sort(key=lambda x: x[0], reverse=True)
        used_t, used_p = set(), set()

        for score, ti, pi, label in candidates:
            if ti in used_t or pi in used_p:
                continue
            used_t.add(ti)
            used_p.add(pi)
            tracker["tp"][label] += 1
            tracker["tot_tp"] += 1

        # Any predicted span not used is a false positive
        for pi, (_, _, p_label) in enumerate(pred_spans):
            if pi not in used_p:
                tracker["fp"][p_label] += 1
                tracker["tot_fp"] += 1

        # Any true span not used is a false negative
        for ti, (_, _, t_label) in enumerate(true_spans):
            if ti not in used_t:
                tracker["fn"][t_label] += 1
                tracker["tot_fn"] += 1
    
    # Scoring definitions
    @staticmethod
    def _score_loose(ps, pe, ts, te):
        overlap = max(0, min(pe, te) - max(ps, ts))
        return 1.0 if overlap > 0 else 0.0

    @staticmethod
    def _score_strict(ps, pe, ts, te):
        return 1.0 if (ps == ts and pe == te) else 0.0

    @staticmethod
    def compute_iou(a_start, a_end, b_start, b_end):
        """Intersection over Union of two character ranges. 1.0 = exact match,
        0.0 = no overlap."""
        intersection = max(0, min(a_end, b_end) - max(a_start, b_start))
        union = (a_end - a_start) + (b_end - b_start) - intersection
        return intersection / union if union > 0 else 0.0

    def evaluate(self, y_true_docs, y_pred_docs):
        """
        y_true_docs: List of lists. Each inner list contains (start, end, label) tuples.
        y_pred_docs: List of lists. Each inner list contains (start, end, label) tuples.
        """
        # Set up stats
        eval_modes = ["loose", "strict"] + [f"iou_{t}" for t in self.iou_thresholds]
        stats = {
            mode: {
                "tp": defaultdict(int), 
                "fp": defaultdict(int), 
                "fn": defaultdict(int),
                "tot_tp": 0, 
                "tot_fp": 0, 
                "tot_fn": 0
            }
            for mode in eval_modes
        }

        # Iterate documents and run greedy matching for each mode
        for true_spans, pred_spans in zip(y_true_docs, y_pred_docs):
            self._match_and_update(stats["loose"], true_spans, pred_spans, self._score_loose)
            self._match_and_update(stats["strict"], true_spans, pred_spans, self._score_strict)
            
            for thresh in self.iou_thresholds:
                self._match_and_update(stats[f"iou_{thresh}"], true_spans, pred_spans, 
                                       self.compute_iou, threshold=thresh)

        # Calculate Results (Per-label and Overall)
        results = {"per_label": {label: {} for label in self.all_labels}, "overall": {}}

        for mode in eval_modes:
            mode_stats = stats[mode]
            label_metrics = []
            
            for label in self.all_labels:
                p, r, f1 = self._compute_metrics(
                    mode_stats["tp"][label], mode_stats["fp"][label], mode_stats["fn"][label]
                )
                results["per_label"][label][mode] = {
                    "p": p, "r": r, "f1": f1,
                    "tp": mode_stats["tp"][label], "fp": mode_stats["fp"][label], "fn": mode_stats["fn"][label]
                }
                label_metrics.append((p, r, f1))
                
            # Compute macro metrics (average of all labels) and micro metrics (across entire dataset)
            micro_p, micro_r, micro_f1 = self._compute_metrics(
                mode_stats["tot_tp"], mode_stats["tot_fp"], mode_stats["tot_fn"]
            )
            
            results["overall"][mode] = {
                "macro_p": np.mean([x[0] for x in label_metrics]),
                "macro_r": np.mean([x[1] for x in label_metrics]),
                "macro_f1": np.mean([x[2] for x in label_metrics]),
                "micro_p": micro_p,
                "micro_r": micro_r,
                "micro_f1": micro_f1
            }

        return results


    def span_recall(self, gold_docs: list, pred_docs: list) -> dict:
        """
        Label-agnostic loose recall: proportion of gold spans (any label) that
        have at least one loosely-overlapping predicted span.
        """
        total, found = 0, 0
        for gold_doc, pred_doc in zip(gold_docs, pred_docs):
            gold_spans = gold_doc.get("needs", []) + gold_doc.get("persons", [])
            pred_spans = pred_doc.get("needs", []) + pred_doc.get("persons", [])
            total += len(gold_spans)
            found += sum(
                any(self._score_loose(g["start"], g["end"], p["start"], p["end"])
                    for p in pred_spans)
                for g in gold_spans
            )
        recall = found / total if total > 0 else 0.0
        return {"found": found, "total": total, "recall": recall}

    def print_report(self, results, title="SPAN LEVEL METRICS"):
        self.logger.info("=" * 80)
        self.logger.info(title)
        self.logger.info("=" * 80)

        headers = ["Label", "L-P", "L-R", "L-F1", "S-P", "S-R", "S-F1"]
        table_rows = []

        # Extract per-label results
        for label in self.all_labels:
            l = results["per_label"][label]["loose"]
            s = results["per_label"][label]["strict"]
            
            # Skip if there are no instances of this label
            if l["tp"] + l["fp"] + l["fn"] == 0:
                continue
                
            table_rows.append([
                label,
                f"{l['p']:.3f}", f"{l['r']:.3f}", f"{l['f1']:.3f}",
                f"{s['p']:.3f}", f"{s['r']:.3f}", f"{s['f1']:.3f}"
            ])

        # Separate the individual labels from the aggregate scores
        table_rows.append(["-" * 30, "-" * 7, "-" * 7, "-" * 7, "-" * 7, "-" * 7, "-" * 7])

        # Extract overall results
        ov_l = results["overall"]["loose"]
        ov_s = results["overall"]["strict"]

        table_rows.append([
            "MACRO",
            f"{ov_l['macro_p']:.4f}", f"{ov_l['macro_r']:.4f}", f"{ov_l['macro_f1']:.4f}",
            f"{ov_s['macro_p']:.4f}", f"{ov_s['macro_r']:.4f}", f"{ov_s['macro_f1']:.4f}"
        ])
        
        table_rows.append([
            "MICRO",
            f"{ov_l['micro_p']:.4f}", f"{ov_l['micro_r']:.4f}", f"{ov_l['micro_f1']:.4f}",
            f"{ov_s['micro_p']:.4f}", f"{ov_s['micro_r']:.4f}", f"{ov_s['micro_f1']:.4f}"
        ])

        # Print with tabulate
        self.logger.info("\n"+tabulate(table_rows, headers=headers, tablefmt="psql", stralign="left", numalign="left"))


class RelationEvaluator:
    """
    Evaluator for relation extraction. Scores predicted (need_id, person_id)
    pairs against gold relations using unordered pair-level exact match.

    Gold relations are matched bidirectionally.
    """

    def __init__(self, logger):
        self.logger = logger

    def _load_gold_relations(self, doc: dict) -> set:
        valid = set()
        for rel in doc.get("relations", []):
            a, b = str(rel["from"]).strip(), str(rel["to"]).strip()
            valid.add(frozenset((a, b)))
        return valid

    def _compute_metrics(self, tp: int, fp: int, fn: int):
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        return p, r, f1

    def evaluate(self, val_records: list, predict_fn, gold_fn=None) -> dict:
        """
        predict_fn(doc) -> set of (need_id, person_id) pairs.
        gold_fn(doc)   -> set of pairs (optional; defaults to _load_gold_relations).
        Returns {precision, recall, f1, tp, fp, fn}.
        """
        total_tp, total_fp, total_fn = 0, 0, 0
        _gold_fn = gold_fn or self._load_gold_relations

        for doc in val_records:
            gold = {frozenset(pair) for pair in _gold_fn(doc)}
            pred = {frozenset(pair) for pair in predict_fn(doc)}

            tp = len(gold & pred)
            fp = len(pred - gold)
            fn = len(gold - pred)
            total_tp += tp
            total_fp += fp
            total_fn += fn

        p, r, f1 = self._compute_metrics(total_tp, total_fp, total_fn)
        return {"precision": p, "recall": r, "f1": f1,
                "tp": total_tp, "fp": total_fp, "fn": total_fn}

    def print_report(self, results: dict, title: str = "RELATION EXTRACTION"):
        self.logger.info(f"=== {title} ===")
        
        headers = ["Precision", "Recall", "F1", "TP", "FP", "FN"]
        # Format the overall metrics
        rows = [[
            f"{results['precision']:.4f}", 
            f"{results['recall']:.4f}", 
            f"{results['f1']:.4f}", 
            results['tp'], 
            results['fp'], 
            results['fn']
        ]]
        
        self.logger.info("\n" + tabulate(rows, headers=headers, tablefmt="psql", stralign="left", numalign="left"))