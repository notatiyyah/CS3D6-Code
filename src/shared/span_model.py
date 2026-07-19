"""
Shared span-classifier pieces used by both train_span_v3.py and eval_span_v3.py,
so the model architecture and candidate-generation logic can't drift between
training and evaluation.
"""

from collections import defaultdict

import torch
import torch.nn as nn
from transformers import AutoModel

def spans_overlap(a_start, a_end, b_start, b_end):
    """True if spans overlap (excluding adjacency)."""
    return max(a_start, b_start) < min(a_end, b_end)

class SpanClassifier(nn.Module):
    def __init__(self, base_model: str, num_labels: int, class_weight=None):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(base_model)
        # +1 for the ("no entity") class
        self.classifier = nn.Linear(self.backbone.config.hidden_size * 2, num_labels + 1)
        self.class_weight = class_weight

    def forward(self, input_ids, attention_mask, candidate_spans, labels=None):
        hidden_states = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[0]
        start_vecs = hidden_states[candidate_spans[:, 0]]
        end_vecs = hidden_states[candidate_spans[:, 1] - 1]
        logits = self.classifier(torch.cat([start_vecs, end_vecs], dim=1))
        loss = nn.CrossEntropyLoss(weight=self.class_weight)(logits, labels) if labels is not None else None
        return {"loss": loss, "logits": logits} if loss is not None else {"logits": logits}


def generate_candidates(offsets, max_size: int):
    """Return (start_idx, end_idx) candidates over real token positions only."""
    real_positions = [i for i, (s, e) in enumerate(offsets) if s != e]
    candidates = []
    for size in range(1, max_size + 1):
        for i in range(len(real_positions) - size + 1):
            window = real_positions[i:i + size]
            if window[-1] - window[0] == size - 1:
                candidates.append((window[0], window[-1] + 1))
    return candidates