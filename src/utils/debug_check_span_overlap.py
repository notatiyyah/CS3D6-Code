import json
from collections import defaultdict


DATASET_PATH = "data/output/gold_standard.json"


def spans_overlap(a, b):
    """True if spans overlap (excluding adjacency)."""
    return max(a["start"], b["start"]) < min(a["end"], b["end"])


def overlap_type(a, b):
    """
    Classify overlap relationship.
    """
    if a["start"] == b["start"] and a["end"] == b["end"]:
        return "identical"

    # One span fully contains the other
    if (
        (a["start"] <= b["start"] and a["end"] >= b["end"])
        or
        (b["start"] <= a["start"] and b["end"] >= a["end"])
    ):
        return "nested"

    return "partial"


def make_spans(record):
    """
    Normalise annotation format into one list.
    """
    spans = []

    for need in record.get("needs", []):
        spans.append({
            **need,
            "category": "need",
        })

    for person in record.get("entities", record.get("people", [])):
        spans.append({
            **person,
            "category": "person",
        })

    return spans


def add_example(store, example, limit=10):
    if len(store) < limit:
        store.append(example)


def analyse_overlaps(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    stats = defaultdict(int)
    examples = defaultdict(list)

    total_spans = 0
    spans_with_overlap = set()

    print("=" * 60)
    print("OVERLAP ANALYSIS")
    print("=" * 60)
    print(f"Scanning: {json_path}\n")


    for record in records:
        text = record.get("text", "")
        spans = make_spans(record)

        total_spans += len(spans)

        for i, span1 in enumerate(spans):
            for span2 in spans[i + 1:]:

                if not spans_overlap(span1, span2):
                    continue

                spans_with_overlap.add(
                    (record["id"], i)
                )

                category_pair = tuple(
                    sorted(
                        [
                            span1["category"],
                            span2["category"],
                        ]
                    )
                )

                stats[
                    f"{category_pair[0]}-{category_pair[1]}"
                ] += 1


                relationship = overlap_type(
                    span1,
                    span2
                )

                stats[
                    f"overlap_{relationship}"
                ] += 1


                example = {
                    "text": text,
                    "span1": text[
                        span1["start"]:span1["end"]
                    ],
                    "span2": text[
                        span2["start"]:span2["end"]
                    ],
                    "label1": span1.get("label"),
                    "label2": span2.get("label"),
                    "categories": category_pair,
                }

                add_example(
                    examples[
                        f"overlap_{relationship}"
                    ],
                    example,
                )


    overlap_percentage = (
        len(spans_with_overlap) / total_spans * 100
        if total_spans
        else 0
    )


    print("--- Global Statistics ---")
    print(f"Total spans: {total_spans}")
    print(
        f"Spans involved in overlap: "
        f"{len(spans_with_overlap)} "
        f"({overlap_percentage:.2f}%)"
    )


    print("\n--- Overlap Pair Types ---")

    for key in [
        "need-need",
        "person-person",
        "need-person",
    ]:
        print(
            f"{key}: {stats[key]}"
        )


    print("\n--- Overlap Geometry ---")

    for key in [
        "overlap_identical",
        "overlap_nested",
        "overlap_partial",
    ]:
        print(
            f"{key.replace('overlap_', '').title()}: "
            f"{stats[key]}"
        )


    print("\n--- Examples ---")

    for overlap_kind, sample_list in examples.items():

        print(
            f"\n{overlap_kind.replace('_', ' ').title()}"
        )

        for ex in sample_list:
            print(
                f"  '{ex['span1']}' "
                f"({ex['label1']}) <-> "
                f"'{ex['span2']}' "
                f"({ex['label2']})"
            )


analyse_overlaps(DATASET_PATH)