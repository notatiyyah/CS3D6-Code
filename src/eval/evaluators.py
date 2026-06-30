import numpy as np

from collections import defaultdict
from sklearn.metrics import f1_score, precision_score, recall_score

# IoU thresholds swept for the "in-between" matching mode. 0.0 is
# conceptually equivalent to loose; 1.0 is conceptually equivalent to
# strict (only exact-boundary matches have IoU=1.0).
IOU_THRESHOLDS = (0.3, 0.5, 0.7, 0.9)


def spans_overlap(a_start, a_end, b_start, b_end):
    """True if spans overlap (excluding adjacency)."""
    return max(a_start, b_start) < min(a_end, b_end)


def compute_iou(a_start, a_end, b_start, b_end):
    """Intersection over Union of two character ranges. 1.0 = exact match,
    0.0 = no overlap."""
    intersection = max(0, min(a_end, b_end) - max(a_start, b_start))
    union = (a_end - a_start) + (b_end - b_start) - intersection
    return intersection / union if union > 0 else 0.0


class SpanEvaluator():
    """
    Evaluator for span-level NER extraction (Loose, Strict, and IoU-thresholded
    matching).

    Loose: any character overlap + same label counts as a match.
    Strict: exact boundary match (start, end) + same label.
    IoU: a tunable middle ground (see IOU_THRESHOLDS) - requires the overlap
    fraction (intersection/union) of predicted and gold spans to clear a
    threshold. IoU=1.0 is equivalent to strict; very low thresholds approach
    loose. Swept across multiple thresholds since no single cutoff is
    obviously "correct" — different thresholds answer different questions
    about how close the model's boundaries are to gold.

    NOTE on matching order: loose/strict use first-match-wins by list order
    (whichever true span comes first in iteration order claims a prediction).
    IoU matching uses best-IoU-first instead (each gold span is matched to
    its highest-IoU available prediction before moving to the next-best
    pair) since with a real numeric overlap measure to rank by, there's no
    reason to settle for an arbitrary first match when a better one is
    available. This means IoU matching is a strictly more careful matching
    procedure than loose/strict, not just a third threshold on the same
    matching logic.
    """

    def __init__(self, all_labels, logger):
        self.all_labels = sorted(list(all_labels))
        self.logger = logger

    def _compute_metrics(self, tp, fp, fn):
        p = tp / (tp + fp) if tp + fp else 0
        r = tp / (tp + fn) if tp + fn else 0
        f1 = 2 * p * r / (p + r) if p + r else 0

        return p, r, f1

    def evaluate(self, y_true_docs, y_pred_docs):
        """
        y_true_docs: List of lists. Each inner list contains (start, end, label) tuples.
        y_pred_docs: List of lists. Each inner list contains (start, end, label) tuples.
        """
        # Initialise counters for loose matching, strict matching, and each IoU threshold
        tp_l, fp_l, fn_l = defaultdict(int), defaultdict(int), defaultdict(int)
        tp_s, fp_s, fn_s = defaultdict(int), defaultdict(int), defaultdict(int)
        tp_iou = {t: defaultdict(int) for t in IOU_THRESHOLDS}
        fp_iou = {t: defaultdict(int) for t in IOU_THRESHOLDS}
        fn_iou = {t: defaultdict(int) for t in IOU_THRESHOLDS}

        tot_tp_l, tot_fp_l, tot_fn_l = 0, 0, 0
        tot_tp_s, tot_fp_s, tot_fn_s = 0, 0, 0
        tot_tp_iou = {t: 0 for t in IOU_THRESHOLDS}
        tot_fp_iou = {t: 0 for t in IOU_THRESHOLDS}
        tot_fn_iou = {t: 0 for t in IOU_THRESHOLDS}

        # Loop through predictions (per ID) & update accordingly
        for true_spans, pred_spans in zip(y_true_docs, y_pred_docs):
            # --- Loose Matching ---
            used_true_loose = set()
            for p_start, p_end, p_label in pred_spans:
                matched = False
                for i, (t_start, t_end, t_label) in enumerate(true_spans):
                    if i in used_true_loose: continue
                    if p_label == t_label and spans_overlap(p_start, p_end, t_start, t_end):
                        tp_l[p_label] += 1
                        tot_tp_l += 1
                        used_true_loose.add(i)
                        matched = True
                        break
                if not matched:
                    fp_l[p_label] += 1
                    tot_fp_l += 1
            for i, (t_start, t_end, t_label) in enumerate(true_spans):
                if i not in used_true_loose:
                    fn_l[t_label] += 1
                    tot_fn_l += 1

            # --- Strict Matching ---
            used_true_strict = set()
            for p_start, p_end, p_label in pred_spans:
                matched = False
                for i, (t_start, t_end, t_label) in enumerate(true_spans):
                    if i in used_true_strict: continue
                    if p_label == t_label and p_start == t_start and p_end == t_end:
                        tp_s[p_label] += 1
                        tot_tp_s += 1
                        used_true_strict.add(i)
                        matched = True
                        break
                if not matched:
                    fp_s[p_label] += 1
                    tot_fp_s += 1
            for i, (t_start, t_end, t_label) in enumerate(true_spans):
                if i not in used_true_strict:
                    fn_s[t_label] += 1
                    tot_fn_s += 1

            # --- IoU Matching (per threshold), best-IoU-first ---
            for threshold in IOU_THRESHOLDS:
                used_true_iou, used_pred_iou = set(), set()
                candidate_pairs = []
                for pi, (p_start, p_end, p_label) in enumerate(pred_spans):
                    for ti, (t_start, t_end, t_label) in enumerate(true_spans):
                        if p_label != t_label:
                            continue
                        iou = compute_iou(p_start, p_end, t_start, t_end)
                        if iou >= threshold:
                            candidate_pairs.append((iou, ti, pi, p_label))

                candidate_pairs.sort(key=lambda x: x[0], reverse=True)
                for iou, ti, pi, label in candidate_pairs:
                    if ti in used_true_iou or pi in used_pred_iou:
                        continue
                    used_true_iou.add(ti)
                    used_pred_iou.add(pi)
                    tp_iou[threshold][label] += 1
                    tot_tp_iou[threshold] += 1

                for pi, (p_start, p_end, p_label) in enumerate(pred_spans):
                    if pi not in used_pred_iou:
                        fp_iou[threshold][p_label] += 1
                        tot_fp_iou[threshold] += 1
                for ti, (t_start, t_end, t_label) in enumerate(true_spans):
                    if ti not in used_true_iou:
                        fn_iou[threshold][t_label] += 1
                        tot_fn_iou[threshold] += 1

        
        # --- Calculate results per label ---
        results = {"per_label": {}, "overall": {}}
        # Calculate precision, recall, f1 loose and strict for each label
        for label in self.all_labels:
            p_l, r_l, f1_l = self._compute_metrics(tp_l[label], fp_l[label], fn_l[label])
            p_s, r_s, f1_s = self._compute_metrics(tp_s[label], fp_s[label], fn_s[label])

            results["per_label"][label] = {
                "loose": {"p": p_l, "r": r_l, "f1": f1_l, "tp": tp_l[label], "fp": fp_l[label], "fn": fn_l[label]},
                "strict": {"p": p_s, "r": r_s, "f1": f1_s, "tp": tp_s[label], "fp": fp_s[label], "fn": fn_s[label]}
            }

            for threshold in IOU_THRESHOLDS:
                p_i, r_i, f1_i = self._compute_metrics(
                    tp_iou[threshold][label], fp_iou[threshold][label], fn_iou[threshold][label]
                )
                results["per_label"][label][f"iou_{threshold}"] = {
                    "p": p_i, "r": r_i, "f1": f1_i,
                    "tp": tp_iou[threshold][label], "fp": fp_iou[threshold][label], "fn": fn_iou[threshold][label],
                }


        # --- Calculate overall results ---
        loose_stats = [m["loose"] for m in results["per_label"].values()]
        strict_stats = [m["strict"] for m in results["per_label"].values()]
        
        # Micro p, r, f1 are calculated over the whole dataset. 
        # Macros are calculated by averaging each label's stats.
        micro_stats_loose = self._compute_metrics(tot_tp_l, tot_fp_l, tot_fn_l)
        micro_stats_strict = self._compute_metrics(tot_tp_s, tot_fp_s, tot_fn_s)

        results["overall"]["loose"] = {
            "macro_p": np.mean([m["p"] for m in loose_stats]),
            "macro_r": np.mean([m["r"] for m in loose_stats]),
            "macro_f1": np.mean([m["f1"] for m in loose_stats]),
            "micro_p": micro_stats_loose[0],
            "micro_r": micro_stats_loose[1],
            "micro_f1": micro_stats_loose[2]
        }
        
        results["overall"]["strict"] = {
            "macro_p": np.mean([m["p"] for m in strict_stats]),
            "macro_r": np.mean([m["r"] for m in strict_stats]),
            "macro_f1": np.mean([m["f1"] for m in strict_stats]),
            "micro_p": micro_stats_strict[0],
            "micro_r": micro_stats_strict[1],
            "micro_f1": micro_stats_strict[2]
        }

        for threshold in IOU_THRESHOLDS:
            iou_stats = [m[f"iou_{threshold}"] for m in results["per_label"].values()]
            micro_stats_iou = self._compute_metrics(tot_tp_iou[threshold], tot_fp_iou[threshold], tot_fn_iou[threshold])
            results["overall"][f"iou_{threshold}"] = {
                "macro_p": np.mean([m["p"] for m in iou_stats]),
                "macro_r": np.mean([m["r"] for m in iou_stats]),
                "macro_f1": np.mean([m["f1"] for m in iou_stats]),
                "micro_p": micro_stats_iou[0],
                "micro_r": micro_stats_iou[1],
                "micro_f1": micro_stats_iou[2],
            }
        
        return results


    def print_report(self, results, title="SPAN LEVEL METRICS"):
        self.logger.info("=" * 130)
        self.logger.info(title)
        self.logger.info("=" * 130)
        self.logger.info(
            "%-45s | %-8s %-8s %-8s | %-8s %-8s %-8s",
            "Label", "L-P", "L-R", "L-F1", "S-P", "S-R", "S-F1"
        )
        self.logger.info("-" * 130)

        for label in self.all_labels:
            l = results["per_label"][label]["loose"]
            s = results["per_label"][label]["strict"]
            if l["tp"] + l["fp"] + l["fn"] == 0:
                continue
            self.logger.info(
                "%-45s | %-8.3f %-8.3f %-8.3f | %-8.3f %-8.3f %-8.3f",
                label, l["p"], l["r"], l["f1"], s["p"], s["r"], s["f1"]
            )

        self.logger.info("-" * 130)
        ov_l = results["overall"]["loose"]
        ov_s = results["overall"]["strict"]
        self.logger.info(
            "%-45s | %-8.4f %-8.4f %-8.4f | %-8.4f %-8.4f %-8.4f",
            "MACRO", ov_l["macro_p"], ov_l["macro_r"], ov_l["macro_f1"],
            ov_s["macro_p"], ov_s["macro_r"], ov_s["macro_f1"]
        )
        self.logger.info(
            "%-45s | %-8.4f %-8.4f %-8.4f | %-8.4f %-8.4f %-8.4f",
            "MICRO", ov_l["micro_p"], ov_l["micro_r"], ov_l["micro_f1"],
            ov_s["micro_p"], ov_s["micro_r"], ov_s["micro_f1"]
        )
