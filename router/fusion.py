"""
Fusion: several ModelOutputs -> one answer string.

Deliberately simple and deliberately honest. Two rules:

  1. Fusion never invents content. It selects, orders and joins text the
     models actually produced. The moment a fuser paraphrases, you have
     added a hallucination surface with no model behind it.
  2. Disagreement is reported, not hidden. If two models say different
     things, the user sees that. A system that silently picks one and
     projects false confidence is worse than one that says "these two
     disagree".
"""

from __future__ import annotations

from contracts import Evidence, FusionStrategy, ModelOutput
from eval.metrics import normalise


def _dedupe_evidence(outputs: list[ModelOutput]) -> list[Evidence]:
    """Merge evidence, dropping exact duplicate boxes from different models."""
    seen: set[tuple] = set()
    merged: list[Evidence] = []
    for out in outputs:
        for ev in out.evidence:
            key = (ev.image_index, ev.bbox, ev.kind.value)
            if key in seen:
                continue
            seen.add(key)
            merged.append(ev)
    return merged


def _agreement(outputs: list[ModelOutput]) -> float:
    """Fraction of output pairs whose normalised text matches."""
    texts = [normalise(o.answer) for o in outputs if o.answer]
    if len(texts) < 2:
        return 1.0
    pairs = agree = 0
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            pairs += 1
            if texts[i] == texts[j]:
                agree += 1
    return agree / pairs if pairs else 1.0


# --------------------------------------------------------------------------
# Strategies
# --------------------------------------------------------------------------

def fuse_best(outputs: list[ModelOutput]) -> tuple[str, float]:
    """Highest-confidence answer wins. Others are noted as support."""
    ranked = sorted(outputs, key=lambda o: -o.confidence)
    top = ranked[0]
    if len(ranked) == 1:
        return top.answer, top.confidence

    others = [o for o in ranked[1:] if o.answer]
    if others:
        support = "; ".join(f"{o.model_id}: {o.answer}" for o in others)
        return f"{top.answer}\n\nAlso considered — {support}", top.confidence
    return top.answer, top.confidence


def fuse_concat(outputs: list[ModelOutput]) -> tuple[str, float]:
    """Every output in order, each attributed. Used for multi-model tasks
    where the models do different jobs rather than the same job."""
    parts = [o.answer.strip() for o in outputs if o.answer.strip()]
    if not parts:
        return "No model produced an answer.", 0.0
    text = " ".join(parts)
    mean_conf = sum(o.confidence for o in outputs) / len(outputs)
    return text, round(mean_conf, 3)


def fuse_vote(outputs: list[ModelOutput]) -> tuple[str, float]:
    """Agreement raises confidence, disagreement lowers it and is reported."""
    ranked = sorted(outputs, key=lambda o: -o.confidence)
    top = ranked[0]
    agree = _agreement(outputs)

    if agree >= 0.99:
        return top.answer, round(min(top.confidence * 1.15, 1.0), 3)

    dissent = [o for o in ranked[1:] if normalise(o.answer) != normalise(top.answer)]
    note = "; ".join(f"{o.model_id} says: {o.answer}" for o in dissent[:2])
    return (f"{top.answer}\n\nModels disagree — {note}",
            round(top.confidence * agree, 3))


_STRATEGIES = {
    FusionStrategy.BEST: fuse_best,
    FusionStrategy.CONCAT: fuse_concat,
    FusionStrategy.VOTE: fuse_vote,
}


def fuse(outputs: list[ModelOutput],
         strategy: FusionStrategy = FusionStrategy.BEST
         ) -> tuple[str, float, list[Evidence]]:
    """Returns (text, confidence, merged evidence)."""
    if not outputs:
        return "No model produced an answer.", 0.0, []

    # A veto means the inputs themselves are unsound. Every other model in
    # the plan produced a confident answer about nonsense, so showing them
    # alongside would be actively misleading.
    vetoed = [o for o in outputs if o.veto]
    if vetoed:
        top = max(vetoed, key=lambda o: o.confidence)
        return top.answer, top.confidence, _dedupe_evidence([top])

    fn = _STRATEGIES.get(strategy, fuse_best)
    text, conf = fn(outputs)
    return text, conf, _dedupe_evidence(outputs)
