"""
Metrics.

Kept separate from the harness so the numbers are auditable. If an evaluator
asks how you computed accuracy, you open this file rather than explaining.

Two things worth defending out loud in the pitch:

  - Normalisation is documented, not silently generous. `normalise` strips
    articles, punctuation and case. It does NOT do synonym matching, because
    the moment you map "woods" to "forest" your accuracy becomes a function
    of your synonym table rather than your model.
  - Multi-label uses micro-F1, not accuracy. On BigEarthNet, predicting "no
    labels" for everything scores well on plain accuracy and is worthless.
"""

from __future__ import annotations

import re
import string
from dataclasses import dataclass, field

_ARTICLES = {"a", "an", "the"}


def normalise(text: str) -> str:
    """Lowercase, strip punctuation and articles, collapse whitespace.

    Known limitation: a label that IS an article normalises to the empty
    string. Harmless for land-cover vocabularies, would matter in another
    domain. Covered by tests/test_week2.py so it stays visible.
    """
    text = text.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = [t for t in text.split() if t not in _ARTICLES]
    return " ".join(tokens)


def to_label_set(text: str) -> set[str]:
    """Comma or semicolon separated label list -> normalised set."""
    parts = re.split(r"[,;]", text)
    return {normalise(p) for p in parts if normalise(p)}


# --------------------------------------------------------------------------
# Scores
# --------------------------------------------------------------------------

def exact_match(pred: str, gold: str) -> float:
    return 1.0 if normalise(pred) == normalise(gold) else 0.0

def contains_match(pred: str, gold: str) -> float:
    """Looser: gold appears inside the prediction.

    Report this ALONGSIDE exact match, never instead of it. A model that
    rambles can score well here while being useless.
    """
    g, p = normalise(gold), normalise(pred)
    return 1.0 if g and g in p else 0.0


def multilabel_prf(pred: str, gold: str) -> tuple[float, float, float]:
    """Per-sample precision, recall, F1 over label sets."""
    p, g = to_label_set(pred), to_label_set(gold)
    if not p and not g:
        return 1.0, 1.0, 1.0
    tp = len(p & g)
    prec = tp / len(p) if p else 0.0
    rec = tp / len(g) if g else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1


# --------------------------------------------------------------------------
# Accumulator
# --------------------------------------------------------------------------

@dataclass
class TaskScore:
    task: str
    n: int = 0
    exact: float = 0.0
    contains: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    latency_ms: float = 0.0
    examples: list[dict] = field(default_factory=list)

    def add(self, pred: str, gold: str, latency_ms: float = 0.0,
            multilabel: bool = False, keep_example: bool = False) -> None:
        self.n += 1
        self.exact += exact_match(pred, gold)
        self.contains += contains_match(pred, gold)
        self.latency_ms += latency_ms
        if multilabel:
            p, r, f = multilabel_prf(pred, gold)
            self.precision += p
            self.recall += r
            self.f1 += f
        if keep_example and len(self.examples) < 5:
            self.examples.append({"pred": pred[:120], "gold": gold[:120]})

    def summary(self) -> dict[str, float | int | str]:
        d = max(self.n, 1)
        return {
            "task": self.task,
            "n": self.n,
            "exact_match": round(self.exact / d, 4),
            "contains": round(self.contains / d, 4),
            "precision": round(self.precision / d, 4),
            "recall": round(self.recall / d, 4),
            "f1": round(self.f1 / d, 4),
            "mean_latency_ms": round(self.latency_ms / d, 1),
        }


def markdown_table(rows: list[dict], columns: list[str]) -> str:
    """Rows -> a markdown table you can paste straight into the deck."""
    if not rows:
        return "_no results_"
    head = "| " + " | ".join(columns) + " |"
    rule = "|" + "|".join("---" for _ in columns) + "|"
    body = []
    for r in rows:
        cells = []
        for c in columns:
            v = r.get(c, "")
            cells.append(f"{v:.1%}" if isinstance(v, float) and 0 <= v <= 1
                         else str(v))
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([head, rule, *body])
