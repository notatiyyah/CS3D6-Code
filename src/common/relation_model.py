"""
Shared pieces used by train_relation_extraction.py, eval_relation_extraction.py,
and eval_relation_closest_match_baseline.py, so marker injection and scoring
logic can't drift between training, model eval, and the baseline.
"""

SPECIAL_TOKENS = ["[N_START]", "[N_END]", "[P_START]", "[P_END]"]


def insert_markers(text: str, need: dict, person: dict) -> str:
    """Injects marker tokens around the specific spans we are classifying.
    Spans are inserted back-to-front so earlier insertions don't shift the
    offsets of spans not yet processed."""
    spans = sorted([
        (need["start"], need["end"], "[N_START]", "[N_END]"),
        (person["start"], person["end"], "[P_START]", "[P_END]"),
    ], key=lambda x: x[0], reverse=True)

    marked_text = text
    for start, end, t_start, t_end in spans:
        marked_text = marked_text[:start] + t_start + marked_text[start:end] + t_end + marked_text[end:]

    return marked_text


def load_gold_relations(doc: dict) -> set:
    """Gold relation pairs as (need_id, person_id), matching both directions
    as 'linked' since source annotations were inconsistent about from/to
    direction — same safety net used at training time."""
    valid_relations = set()
    for rel in doc.get("relations", []):
        rel_from = str(rel["from"]).strip()
        rel_to = str(rel["to"]).strip()
        valid_relations.add((rel_from, rel_to))
        valid_relations.add((rel_to, rel_from))
    return valid_relations


def compute_pair_metrics(tp: int, fp: int, fn: int):
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1


def build_entity_lookup(doc: dict) -> dict:
    """Map every need/person ID in a document to its record (with text/label),
    so wrong pairs can be reported with readable context, not just bare IDs."""
    lookup = {}
    for need in doc.get("needs", []):
        lookup[str(need["id"]).strip()] = {"kind": "need", **need}
    for person in doc.get("persons", []):
        lookup[str(person["id"]).strip()] = {"kind": "person", **person}
    return lookup


def describe_pair(doc: dict, pair: frozenset, lookup: dict) -> dict:
    """Render an unordered (need_id, person_id) pair with the actual span
    text pulled from the document, for human-readable failure reports."""
    ids = list(pair)
    entities = []
    for entity_id in ids:
        record = lookup.get(entity_id)
        if record is None:
            entities.append({"id": entity_id, "text": "<unknown id>"})
            continue
        span_text = doc["text"][record["start"]:record["end"]]
        entities.append({
            "id": entity_id,
            "text": span_text,
            "kind": record["kind"],
            "label": record.get("label"),
        })
    return {"entities": entities}


def score_documents_with_detail(val_records, predict_fn, logger):
    """Same as score_documents, but additionally returns the actual wrong
    pairs (fp/fn) per document with readable text, for failure inspection.
    Heavier to compute/store than score_documents — use that for routine
    eval, this only when you need to see *what* went wrong, not just *how
    much*."""
    total_tp, total_fp, total_fn = 0, 0, 0
    per_doc_results = []

    for doc in val_records:
        gold_relations = load_gold_relations(doc)
        predicted_pairs = predict_fn(doc)
        lookup = build_entity_lookup(doc)

        gold_unordered = {frozenset(pair) for pair in gold_relations}
        pred_unordered = {frozenset(pair) for pair in predicted_pairs}

        tp_pairs = gold_unordered & pred_unordered
        fp_pairs = pred_unordered - gold_unordered
        fn_pairs = gold_unordered - pred_unordered

        tp, fp, fn = len(tp_pairs), len(fp_pairs), len(fn_pairs)
        total_tp += tp
        total_fp += fp
        total_fn += fn

        n_gold = len(gold_unordered)
        _, _, doc_f1 = compute_pair_metrics(tp, fp, fn)

        per_doc_results.append({
            "doc_id": doc.get("id"),
            "tp": tp, "fp": fp, "fn": fn,
            "n_gold_relations": n_gold,
            "f1": doc_f1,
            "false_positives": [describe_pair(doc, pair, lookup) for pair in fp_pairs],
            "false_negatives": [describe_pair(doc, pair, lookup) for pair in fn_pairs],
        })

    precision, recall, f1 = compute_pair_metrics(total_tp, total_fp, total_fn)
    logger.info(
        "Pair-level exact match: precision=%.4f recall=%.4f f1=%.4f (tp=%s fp=%s fn=%s)",
        precision, recall, f1, total_tp, total_fp, total_fn,
    )

    return {
        "overall": {
            "precision": precision, "recall": recall, "f1": f1,
            "tp": total_tp, "fp": total_fp, "fn": total_fn,
        },
        "per_doc": per_doc_results,
    }


def score_documents(val_records, predict_fn, logger):
    """Run predict_fn(doc) -> set of (need_id, person_id) over every document,
    score against gold relations (unordered, pair-level exact match), and
    return the same {overall, per_doc} shape used by every eval script in
    this project."""
    total_tp, total_fp, total_fn = 0, 0, 0
    per_doc_results = []

    for doc in val_records:
        gold_relations = load_gold_relations(doc)
        predicted_pairs = predict_fn(doc)

        gold_unordered = {frozenset(pair) for pair in gold_relations}
        pred_unordered = {frozenset(pair) for pair in predicted_pairs}

        tp = len(gold_unordered & pred_unordered)
        fp = len(pred_unordered - gold_unordered)
        fn = len(gold_unordered - pred_unordered)

        total_tp += tp
        total_fp += fp
        total_fn += fn

        per_doc_results.append({"doc_id": doc.get("id"), "tp": tp, "fp": fp, "fn": fn})

    precision, recall, f1 = compute_pair_metrics(total_tp, total_fp, total_fn)
    logger.info(
        "Pair-level exact match: precision=%.4f recall=%.4f f1=%.4f (tp=%s fp=%s fn=%s)",
        precision, recall, f1, total_tp, total_fp, total_fn,
    )

    return {
        "overall": {
            "precision": precision, "recall": recall, "f1": f1,
            "tp": total_tp, "fp": total_fp, "fn": total_fn,
        },
        "per_doc": per_doc_results,
    }