import numpy as np

from collections import defaultdict
from sklearn.metrics import f1_score, precision_score, recall_score

class Evaluator:
    def __init__(self, all_labels, logger):
        self.all_labels = sorted(list(all_labels))
        self.logger = logger

    def _compute_metrics(self, tp, fp, fn):
        p = tp / (tp + fp) if tp + fp else 0
        r = tp / (tp + fn) if tp + fn else 0
        f1 = 2 * p * r / (p + r) if p + r else 0

        return p, r, f1

class DocLevelEvaluator(Evaluator):
    """
    Evaluator for document-level multi-label classification.
    """
    def __init__(self, all_labels, logger):
        Evaluator.__init__(self, all_labels, logger)
        self.label2id = {lbl: i for i, lbl in enumerate(self.all_labels)}

    def _to_binary_matrix(self, docs_labels):
        # TODO: Replace with utisl one?
        """Converts a list of label lists into a binary matrix."""
        mat = np.zeros((len(docs_labels), len(self.all_labels)), dtype=int)

        for i, labels in enumerate(docs_labels):
            for lbl in labels:
                if lbl in self.label2id:
                    mat[i, self.label2id[lbl]] = 1
        return mat

    def evaluate(self, y_true_lists, y_pred_lists):
        """
        Calculates all document-level metrics.
        y_true_lists: List of lists containing true string labels per document.
        y_pred_lists: List of lists containing predicted string labels per document.
        """
        y_true = self._to_binary_matrix(y_true_lists)
        y_pred = self._to_binary_matrix(y_pred_lists)

        metrics = {
            "overall": {
                "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
                "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
                "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
                "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
                "micro_precision": float(precision_score(y_true, y_pred, average="micro", zero_division=0)),
                "micro_recall": float(recall_score(y_true, y_pred, average="micro", zero_division=0)),
            },
            "per_label": {}
        }

        for i, label in enumerate(self.all_labels):
            tp = np.sum((y_true[:, i] == 1) & (y_pred[:, i] == 1))
            fp = np.sum((y_true[:, i] == 0) & (y_pred[:, i] == 1))
            fn = np.sum((y_true[:, i] == 1) & (y_pred[:, i] == 0))

            p, r, f1 = self._compute_metrics(tp, fp, fn)

            metrics["per_label"][label] = {
                "precision": float(p),
                "recall": float(r),
                "f1": float(f1),
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
            }

        return metrics

    def print_report(self, metrics, title="DOCUMENT-LEVEL CLASSIFICATION METRICS"):
        """Standardized output for the evaluation results."""
        # Title
        self.logger.info("=" * 120)
        self.logger.info(title)
        self.logger.info("=" * 120)

        # Header
        self.logger.info(
            "%-45s | %-10s | %-10s | %-10s | %-5s | %-5s | %-5s",
            "Label",
            "Precision",
            "Recall",
            "F1",
            "TP",
            "FP",
            "FN"
        )

        # Sort by F1 & output table
        for label, m in sorted(metrics["per_label"].items(),
                               key=lambda x: x[1]["f1"],
                               reverse=True):
            if m["tp"] + m["fp"] + m["fn"] > 0:
                self.logger.info(
                    "%-45s | %-10.3f | %-10.3f | %-10.3f | %-5s | %-5s | %-5s",
                    label,
                    m["precision"],
                    m["recall"],
                    m["f1"],
                    m["tp"],
                    m["fp"],
                    m["fn"],
                )

        # Show overalll metrics
        overall = metrics["overall"]
        self.logger.info("-" * 120)
        self.logger.info("Macro F1: %.4f", overall["macro_f1"])
        self.logger.info("Micro F1: %.4f", overall["micro_f1"])
        self.logger.info("=" * 120)


class SpanLevelEvaluator(Evaluator):
    """
    Evaluator for span-level ANs extraction (Loose and Strict matching).
    """

    def evaluate(self, y_true_docs, y_pred_docs):
        """
        y_true_docs: List of lists. Each inner list contains (start, end, label) tuples.
        y_pred_docs: List of lists. Each inner list contains (start, end, label) tuples.
        """
        # Initialise counters for both loose matching & strict matching
        tp_l, fp_l, fn_l = defaultdict(int), defaultdict(int), defaultdict(int)
        tp_s, fp_s, fn_s = defaultdict(int), defaultdict(int), defaultdict(int)

        tot_tp_l, tot_fp_l, tot_fn_l = 0, 0, 0
        tot_tp_s, tot_fp_s, tot_fn_s = 0, 0, 0

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
            for i, (t_start, t_end, t_label) in enumerate(true_span_list):
                if i not in used_true_strict:
                    fn_s[t_label] += 1
                    tot_fn_s += 1

        
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
        
        return results


    def print_report(self, results, title="SPAN LEVEL METRICS"):
        # title
        self.logger.info("=" * 120)
        self.logger.info(title)
        self.logger.info("=" * 120)

        # table header
        self.logger.info(
            "%-45s | %-15s | %-15s | %-10s",
            "Label",
            "Loose F1",
            "Strict F1",
            "Delta"
        )

        # Results per label
        for label in self.all_labels:
            loose = results["per_label"][label]["loose"]
            strict = results["per_label"][label]["strict"]

            if loose["tp"] + loose["fp"] + loose["fn"] > 0:
                self.logger.info(
                    "%-45s | %-15.3f | %-15.3f | %.3f",
                    label,
                    loose["f1"],
                    strict["f1"],
                    strict["f1"] - loose["f1"]
                )
        
        # Overall stats
        ov_l, ov_s = results["overall"]["loose"], results["overall"]["strict"]
        self.logger.info("-" * 120)
        self.logger.info(f"{'Metric':<25} | {'Loose':<25} | {'Strict':<25} | {'Delta'}")
        self.logger.info("-" * 120)
        self.logger.info(f"{'Macro F1':<25} | {ov_l['macro_f1']:<25.4f} | {ov_s['macro_f1']:<25.4f} | {ov_s['macro_f1'] - ov_l['macro_f1']:.4f}")
        self.logger.info(f"{'Micro F1':<25} | {ov_l['micro_f1']:<25.4f} | {ov_s['micro_f1']:<25.4f} | {ov_s['micro_f1'] - ov_l['micro_f1']:.4f}")