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

