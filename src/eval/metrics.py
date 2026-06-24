import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from collections import defaultdict

def spans_overlap(a_start, a_end, b_start, b_end):
    """Utility to check if two character spans overlap."""
    return max(a_start, b_start) < min(a_end, b_end)

class DocLevelEvaluator:
    """
    Centralized evaluator for document-level multi-label classification.
    """
    def __init__(self, all_labels):
        self.all_labels = sorted(list(all_labels))
        self.label2id = {lbl: i for i, lbl in enumerate(self.all_labels)}

    def _to_binary_matrix(self, docs_labels):
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
            
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            
            metrics["per_label"][label] = {
                "precision": float(p), "recall": float(r), "f1": float(f1),
                "tp": int(tp), "fp": int(fp), "fn": int(fn)
            }
            
        return metrics

    def print_report(self, metrics, title="DOCUMENT-LEVEL CLASSIFICATION METRICS"):
        """Standardized console output for the evaluation results."""
        print(f"\n{'='*95}\n{title}\n{'='*95}")
        print(f"{'Label':<45} | {'Precision':<10} | {'Recall':<10} | {'F1':<10} | {'TP':<5} | {'FP':<5} | {'FN':<5}")
        print("-" * 95)
        
        # Sort by F1 descending for easier reading
        sorted_labels = sorted(metrics["per_label"].items(), key=lambda x: x[1]['f1'], reverse=True)
        
        for label, m in sorted_labels:
            if m["tp"] + m["fp"] + m["fn"] > 0:
                print(f"{label:<45} | {m['precision']:<10.3f} | {m['recall']:<10.3f} | {m['f1']:<10.3f} | {m['tp']:<5} | {m['fp']:<5} | {m['fn']:<5}")
        
        print(f"\n{'-'*95}")
        ov = metrics["overall"]
        print(f"{'OVERALL':<45} | {'':<10} | {'':<10} | {'':<10} | {'':<5} | {'':<5} | {'':<5}")
        print(f"{'Macro F1':<45} | {'':<10} | {'':<10} | {ov['macro_f1']:<10.4f} | {'':<5} | {'':<5} | {'':<5}")
        print(f"{'Micro F1':<45} | {'':<10} | {'':<10} | {ov['micro_f1']:<10.4f} | {'':<5} | {'':<5} | {'':<5}")
        print("=" * 95)

class SpanLevelEvaluator:
    """
    Centralized evaluator for span-level entity extraction (Loose and Strict matching).
    """
    def __init__(self, all_labels):
        self.all_labels = sorted(list(all_labels))

    def _compute_metrics(self, tp, fp, fn):
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        return p, r, f1

    def evaluate(self, y_true_docs, y_pred_docs):
        """
        y_true_docs: List of lists. Each inner list contains (start, end, label) tuples.
        y_pred_docs: List of lists. Each inner list contains (start, end, label) tuples.
        """
        tp_l, fp_l, fn_l = defaultdict(int), defaultdict(int), defaultdict(int)
        tp_s, fp_s, fn_s = defaultdict(int), defaultdict(int), defaultdict(int)
        
        tot_tp_l, tot_fp_l, tot_fn_l = 0, 0, 0
        tot_tp_s, tot_fp_s, tot_fn_s = 0, 0, 0

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

        # --- Aggregation ---
        results = {"per_label": {}, "overall": {}}
        
        for label in self.all_labels:
            p_l, r_l, f1_l = self._compute_metrics(tp_l[label], fp_l[label], fn_l[label])
            p_s, r_s, f1_s = self._compute_metrics(tp_s[label], fp_s[label], fn_s[label])
            results["per_label"][label] = {
                "loose": {"p": p_l, "r": r_l, "f1": f1_l, "tp": tp_l[label], "fp": fp_l[label], "fn": fn_l[label]},
                "strict": {"p": p_s, "r": r_s, "f1": f1_s, "tp": tp_s[label], "fp": fp_s[label], "fn": fn_s[label]}
            }

        # Calculate Macros
        valid_l = [m["loose"] for m in results["per_label"].values() if m["loose"]["tp"] + m["loose"]["fp"] + m["loose"]["fn"] > 0]
        valid_s = [m["strict"] for m in results["per_label"].values() if m["strict"]["tp"] + m["strict"]["fp"] + m["strict"]["fn"] > 0]
        
        results["overall"]["loose"] = {
            "macro_p": np.mean([m["p"] for m in valid_l]) if valid_l else 0,
            "macro_r": np.mean([m["r"] for m in valid_l]) if valid_l else 0,
            "macro_f1": np.mean([m["f1"] for m in valid_l]) if valid_l else 0,
            "micro_p": self._compute_metrics(tot_tp_l, tot_fp_l, tot_fn_l)[0],
            "micro_r": self._compute_metrics(tot_tp_l, tot_fp_l, tot_fn_l)[1],
            "micro_f1": self._compute_metrics(tot_tp_l, tot_fp_l, tot_fn_l)[2]
        }
        
        results["overall"]["strict"] = {
            "macro_p": np.mean([m["p"] for m in valid_s]) if valid_s else 0,
            "macro_r": np.mean([m["r"] for m in valid_s]) if valid_s else 0,
            "macro_f1": np.mean([m["f1"] for m in valid_s]) if valid_s else 0,
            "micro_p": self._compute_metrics(tot_tp_s, tot_fp_s, tot_fn_s)[0],
            "micro_r": self._compute_metrics(tot_tp_s, tot_fp_s, tot_fn_s)[1],
            "micro_f1": self._compute_metrics(tot_tp_s, tot_fp_s, tot_fn_s)[2]
        }
        
        return results

    def print_report(self, results, title="SPAN-LEVEL METRICS"):
        print(f"\n{'='*115}\n{title}\n{'='*115}")
        print(f"{'Label':<45} | {'Loose F1':<15} | {'Strict F1':<15} | {'Δ (Drift)':<10}")
        print("-" * 115)
        
        for label in self.all_labels:
            m_l, m_s = results["per_label"][label]["loose"], results["per_label"][label]["strict"]
            if m_l["tp"] + m_l["fp"] + m_l["fn"] > 0:
                print(f"{label:<45} | {m_l['f1']:<15.3f} | {m_s['f1']:<15.3f} | {m_s['f1'] - m_l['f1']:<10.3f}")

        ov_l, ov_s = results["overall"]["loose"], results["overall"]["strict"]
        print(f"\n{'='*115}\nSUMMARY\n{'='*115}")
        print(f"{'Metric':<25} | {'Loose':<25} | {'Strict':<25} | {'Δ Drop'}")
        print("-" * 115)
        print(f"{'Macro F1':<25} | {ov_l['macro_f1']:<25.4f} | {ov_s['macro_f1']:<25.4f} | {ov_s['macro_f1'] - ov_l['macro_f1']:.4f}")
        print(f"{'Micro F1':<25} | {ov_l['micro_f1']:<25.4f} | {ov_s['micro_f1']:<25.4f} | {ov_s['micro_f1'] - ov_l['micro_f1']:.4f}")
        print("=" * 115)
