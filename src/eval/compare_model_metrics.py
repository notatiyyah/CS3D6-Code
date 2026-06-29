"""
Compare eval results across models, even when their result JSONs have
different shapes (flat has one block, hierarchical has three, span
classifiers will have loose/strict).

Add a model by pointing at its file and listing the dotted path(s) to the
{overall, per_label} block(s) you want shown as rows. No path needed if the
file already *is* a {overall, per_label} block at the top level.
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from common.logging import setup_logger
from common.paths import METRICS

LOGGER = setup_logger("eval.compare_models", "compare_models.log")
OVERALL_METRICS = ["macro_f1", "micro_f1", "macro_precision", "macro_recall", "micro_precision", "micro_recall"]


@dataclass
class ResultSource:
    """One row in the comparison. `path` is either:
    - a dotted path to a {overall, per_label} block (e.g. hierarchical's
      "overall_chained" / "parent" / "child_isolated"), or
    - a metric name nested *inside* a top-level {overall, per_label} block
      (e.g. span results' "loose" / "strict").
    Leave as "" if the file's top level already is a {overall, per_label} block.

    `match_type` groups rows for "best per label" comparisons in the per-label
    table — e.g. loose-matched span rows only compete against other loose rows,
    never against strict or doc-level rows. Defaults to "doc" for everything
    that isn't an explicit span loose/strict row."""
    name: str
    file: Path
    path: str = ""
    match_type: str = "doc"


@dataclass
class CompareConfig:
    sources: list
    output_dir: Path = METRICS


# --- Edit this for whichever models/files you want to compare ---
CONFIG = CompareConfig(sources=[
    # ResultSource("flat", METRICS / "doc_flat.20260627_003403.json", path="results"),
    # ResultSource("hierarchical_chained", METRICS / "doc_hierarchical.20260627_004200.json", path="overall_chained"),
    # ResultSource("regex", METRICS / "doc_regex.json"),
    # ResultSource("gemini", METRICS / "doc_gemini.json"),
    # ResultSource("distilbert_loose", METRICS / "span_v3.20260627_155848.json", path="loose"),
    ResultSource("roberta_loose", METRICS / "span.20260628_155640.json", path="loose"),
    # ResultSource("clinicalbert_loose", METRICS / "span.20260628_172459.json", path="loose"),
    ResultSource("deberta_loose", METRICS / "span.20260628_194632.json", path="loose"),
    # ResultSource("regex_loose", METRICS / "span_regex.json", path="loose"),
    # ResultSource("gemini_loose", METRICS / "span_gemini.json", path="loose"),
    # ResultSource("distilbert_strict", METRICS / "span_v3.20260627_155848.json", path="strict"),
    ResultSource("roberta_strict", METRICS / "span.20260628_155640.json", path="strict"),
    # ResultSource("clinicalbert_strict", METRICS / "span.20260628_172459.json", path="strict"),
    ResultSource("deberta_strict", METRICS / "span.20260628_194632.json", path="strict"),
    # ResultSource("regex_strict", METRICS / "span_regex.json", path="strict"),
    # ResultSource("gemini_strict", METRICS / "span_gemini.json", path="strict"),
])


METRIC_KEY_ALIASES = {
    "macro_p": "macro_precision",
    "macro_r": "macro_recall",
    "micro_p": "micro_precision",
    "micro_r": "micro_recall",
}


def normalize_overall_keys(overall: dict) -> dict:
    """Span eval scripts use short-form keys (macro_p/macro_r/micro_p/micro_r);
    doc-level scripts use long-form (macro_precision/macro_recall/...). Normalize
    to long-form so both can sit in the same comparison table."""
    return {METRIC_KEY_ALIASES.get(k, k): v for k, v in overall.items()}


def get_path(data: dict, dotted_path: str) -> dict:
    if not dotted_path:
        return data
    node = data
    for key in dotted_path.split("."):
        node = node[key]
    return node


def load_block(source: ResultSource) -> dict:
    """Three result shapes exist in the wild:

    1. Flat-nested (doc-level flat/hierarchical/gemini/regex): the dotted
       path leads straight to a {overall, per_label} block, e.g.
       data["overall_chained"] = {"overall": {...}, "per_label": {...}}.

    2. Metric-nested (span loose/strict, single threshold): "overall" and
       "per_label" are top-level, and the metric name (e.g. "strict") is
       nested *inside* each of them, e.g. data["overall"]["strict"].

    3. Threshold-swept (span loose/strict across a threshold sweep): the
       top level is a dict of threshold strings ("0.5", "0.7", ...), each
       mapping to a shape-2 block. The path names the metric ("loose" /
       "strict"); the best threshold for that metric (by macro_f1) is
       picked automatically, mirroring eval_flat.py's best-threshold logic.

    Try shape 1 first, then shape 3 (if the top level looks like a
    threshold sweep), then shape 2.
    """
    import json
    with open(source.file, encoding="utf-8") as f:
        data = json.load(f)

    if not source.path:
        block = data
        if "overall" in block and "per_label" in block:
            return {**block, "overall": normalize_overall_keys(block["overall"])}
        raise ValueError(
            f"'{source.name}': no path given but {source.file} top level lacks "
            f"'overall'/'per_label' (got keys: {list(block.keys())})"
        )

    # Shape 1: dotted path leads directly to a {overall, per_label} block.
    try:
        block = get_path(data, source.path)
        if isinstance(block, dict) and "overall" in block and "per_label" in block:
            return {**block, "overall": normalize_overall_keys(block["overall"])}
    except (KeyError, TypeError):
        pass

    # Shape 3: top level is a threshold sweep — auto-pick the best threshold
    # for this metric (path), then recurse into shape 2 on that sub-block.
    if is_threshold_sweep(data):
        metric = source.path
        best_threshold, best_block = pick_best_threshold(data, metric)
        LOGGER.info(
            "'%s': auto-selected threshold=%s (best macro_f1 for metric=%r) from %s",
            source.name, best_threshold, metric, source.file,
        )
        return load_metric_nested_block(source, best_block, metric)

    # Shape 2: path is a metric name nested inside "overall" and each per_label entry.
    if "overall" in data and "per_label" in data:
        return load_metric_nested_block(source, data, source.path)

    raise ValueError(
        f"'{source.name}': path '{source.path}' in {source.file} doesn't match any "
        f"known result shape (got top-level keys: {list(data.keys())})"
    )


def is_threshold_sweep(data: dict) -> bool:
    """True if the top level looks like {"0.5": {...}, "0.7": {...}, ...}
    rather than a single result block."""
    if "overall" in data or "per_label" in data:
        return False
    try:
        [float(k) for k in data.keys()]
        return True
    except (ValueError, AttributeError):
        return False


def pick_best_threshold(data: dict, metric: str):
    """Among threshold sub-blocks, pick the one with the highest macro_f1
    for the given metric (loose/strict)."""
    scored = []
    for threshold_str, block in data.items():
        try:
            f1 = block["overall"][metric]["macro_f1"]
            scored.append((f1, threshold_str, block))
        except KeyError:
            continue
    if not scored:
        raise ValueError(f"No threshold sub-block has overall.{metric}.macro_f1 to compare")
    _, best_threshold, best_block = max(scored, key=lambda x: x[0])
    return best_threshold, best_block


def load_metric_nested_block(source: "ResultSource", data: dict, metric: str) -> dict:
    """Shape 2 extraction: pull `metric` out of data['overall'] and out of
    every entry in data['per_label']."""
    try:
        overall = data["overall"][metric]
    except KeyError:
        raise ValueError(
            f"'{source.name}': '{metric}' not found in overall block "
            f"(got keys: {list(data['overall'].keys())})"
        )
    per_label = {
        label: stats[metric]
        for label, stats in data["per_label"].items()
        if metric in stats
    }
    return {"overall": normalize_overall_keys(overall), "per_label": per_label}


def build_overall_table(blocks: dict) -> pd.DataFrame:
    rows = {name: block["overall"] for name, block in blocks.items()}
    return pd.DataFrame(rows).T[OVERALL_METRICS]


def build_per_label_table(blocks: dict, metric: str = "f1") -> pd.DataFrame:
    rows = {}
    for name, block in blocks.items():
        rows[name] = {label: stats.get(metric) for label, stats in block["per_label"].items()}
    return pd.DataFrame(rows)


BOLD = "\033[1m"
RESET = "\033[0m"


def render_table(df: pd.DataFrame, best_idx_per_col: dict = None, best_col_per_idx: dict = None) -> str:
    """Render a DataFrame as a plain-text table with the chosen 'best' cells
    bolded via ANSI codes, computing column widths from visible text only.

    pandas' own to_string() can't be used here: it measures column width
    including the literal escape-code characters, which inflates the width
    of any column containing a bolded cell and misaligns every other column.
    Padding manually, before applying ANSI codes, avoids that entirely.

    Pass exactly one of best_idx_per_col (col -> winning row label, for
    'best per column') or best_col_per_idx (row label -> winning col, for
    'best per row')."""
    index_label = df.index.name or ""
    headers = [index_label] + list(df.columns)

    def cell_text(idx, col):
        value = df.loc[idx, col]
        return f"{value:.4f}" if pd.notna(value) else "NaN"

    def is_best(idx, col):
        if best_idx_per_col is not None:
            return best_idx_per_col.get(col) == idx
        return best_col_per_idx.get(idx) == col

    # Visible widths only — no ANSI codes involved at this stage.
    col_widths = [max(len(str(idx)) for idx in df.index)] if len(df.index) else [len(index_label)]
    col_widths[0] = max(col_widths[0], len(index_label))
    for col in df.columns:
        width = max([len(str(col))] + [len(cell_text(idx, col)) for idx in df.index])
        col_widths.append(width)

    lines = ["  ".join(h.ljust(w) for h, w in zip(headers, col_widths))]
    for idx in df.index:
        row_cells = [str(idx).ljust(col_widths[0])]
        for col, width in zip(df.columns, col_widths[1:]):
            text = cell_text(idx, col)
            padded = text.ljust(width)  # pad first, using visible width
            if is_best(idx, col):
                # Replace only the visible text portion with bold-wrapped text;
                # the padding spaces stay outside the ANSI codes.
                padded = f"{BOLD}{text}{RESET}" + " " * (width - len(text))
            row_cells.append(padded)
        lines.append("  ".join(row_cells))
    return "\n".join(lines)


def best_per_column(df: pd.DataFrame) -> dict:
    result = {}
    for col in df.columns:
        col_values = df[col]
        if col_values.isna().all():
            continue
        result[col] = col_values.idxmax()
    return result


def best_per_row(df: pd.DataFrame) -> dict:
    result = {}
    for idx in df.index:
        row_values = df.loc[idx]
        if row_values.isna().all():
            continue
        result[idx] = row_values.idxmax()
    return result


def main():
    blocks = {}
    for source in CONFIG.sources:
        try:
            blocks[source.name] = load_block(source)
            LOGGER.info("Loaded '%s' from %s (path=%r)", source.name, source.file, source.path)
        except FileNotFoundError:
            LOGGER.info("Skipping '%s': file not found at %s", source.name, source.file)
        except ValueError as e:
            LOGGER.info("Skipping '%s': %s", source.name, e)

    if not blocks:
        print("No valid result blocks loaded — check CONFIG.sources paths.")
        sys.exit(1)

    overall_df = build_overall_table(blocks)
    print("\n=== OVERALL METRICS ===")
    print(render_table(overall_df, best_idx_per_col=best_per_column(overall_df)))

    per_label_f1 = build_per_label_table(blocks, metric="f1")
    print("\n=== PER-LABEL F1 ===")
    print(render_table(per_label_f1, best_col_per_idx=best_per_row(per_label_f1)))

    CONFIG.output_dir.mkdir(parents=True, exist_ok=True)
    overall_df.to_csv(CONFIG.output_dir / "overall_comparison.csv")
    per_label_f1.to_csv(CONFIG.output_dir / "per_label_f1_comparison.csv")
    LOGGER.info("Saved comparison tables to %s", CONFIG.output_dir)


if __name__ == "__main__":
    main()