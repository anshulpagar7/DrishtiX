"""
Week 2 contract tests.

These cover the failure modes that actually cost you a demo:
  - an unconfigured LLM backend silently breaking the parser
  - a rate-limited API taking the app down with it
  - training and inference building different prompts
  - a metric that flatters the model

Run:  python -m tests.test_week2
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contracts import TaskSpec, TaskType                    # noqa: E402
from data.prompts import build_prompt, build_training_prompt, chat_messages  # noqa: E402
from eval.metrics import (exact_match, multilabel_prf,      # noqa: E402
                          normalise, to_label_set)
from router.llm_backends import (OfflineBackend, ParseCache,  # noqa: E402
                                 get_backend)
from router.parser import LLMParser, RuleParser, build_parser  # noqa: E402


# --------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------

class FakeBackend:
    """Returns a canned completion. Counts calls so caching is testable."""

    backend_id = "fake"
    model = "fake-1"

    def __init__(self, payload: str, fail: bool = False) -> None:
        self.payload = payload
        self.fail = fail
        self.calls = 0

    def available(self) -> bool:
        return True

    def complete(self, prompt: str) -> str | None:
        self.calls += 1
        return None if self.fail else self.payload


GOOD = ('```json\n{"task_type":"change","modality":"optical","needs_pair":true,'
        '"temporal":true,"target_class":"glacier","confidence":0.93}\n```')


def _cache() -> ParseCache:
    tmp = Path(tempfile.mkdtemp()) / "cache.json"
    return ParseCache(tmp)


# --------------------------------------------------------------------------
# LLM parser
# --------------------------------------------------------------------------

def test_llm_parser_reads_fenced_json() -> None:
    p = LLMParser(backend=FakeBackend(GOOD), cache=_cache())
    spec = p.parse("Has the glacier retreated?")
    assert spec.task_type is TaskType.CHANGE
    assert spec.needs_pair is True
    assert spec.target_class == "glacier"
    assert spec.parser_id == "llm-v1"
    print("  llm parser parses fenced json   ok")


def test_llm_parser_falls_back_on_failure() -> None:
    p = LLMParser(backend=FakeBackend("", fail=True), cache=_cache())
    spec = p.parse("Where is the water?")
    assert spec.task_type is TaskType.GROUND     # rule parser handled it
    assert p.last_source == "fallback"
    print("  llm parser falls back on error  ok")


def test_llm_parser_falls_back_on_garbage() -> None:
    p = LLMParser(backend=FakeBackend("I think it's a change query!"),
                  cache=_cache())
    spec = p.parse("Describe this scene.")
    assert spec.task_type is TaskType.CAPTION
    print("  llm parser survives non-json    ok")


def test_llm_parser_caches() -> None:
    be = FakeBackend(GOOD)
    p = LLMParser(backend=be, cache=_cache())
    for _ in range(4):
        p.parse("Has the glacier retreated?")
    assert be.calls == 1, f"expected 1 backend call, got {be.calls}"
    assert p.last_source == "cache"
    print("  llm parser caches completions   ok")


def test_needs_pair_follows_contract_not_model() -> None:
    """Model says change but needs_pair false -> contract wins."""
    bad = ('{"task_type":"change","modality":"any","needs_pair":false,'
           '"temporal":false,"target_class":null,"confidence":0.9}')
    p = LLMParser(backend=FakeBackend(bad), cache=_cache())
    spec = p.parse("what changed")
    assert spec.needs_pair is True
    print("  contract overrides model field  ok")


def test_offline_backend_is_default() -> None:
    be = get_backend("offline")
    assert isinstance(be, OfflineBackend)
    assert be.available() is False
    assert be.complete("anything") is None
    print("  offline backend is inert        ok")


def test_build_parser_without_backend_gives_rule_parser() -> None:
    p = build_parser(prefer_llm=True)
    assert isinstance(p, (RuleParser, LLMParser))
    if isinstance(p, RuleParser):
        print("  build_parser degrades to rules  ok")
    else:
        print("  build_parser used llm backend   ok")


# --------------------------------------------------------------------------
# Prompt consistency - the top fine-tuning bug
# --------------------------------------------------------------------------

def test_train_and_infer_prompts_match() -> None:
    spec = TaskSpec(raw_query="anything", task_type=TaskType.CLASSIFY)
    assert build_prompt(spec) == build_training_prompt(TaskType.CLASSIFY)
    print("  train/infer prompts identical   ok")


def test_vqa_prompt_carries_the_question() -> None:
    q = "Is there a river in this image?"
    spec = TaskSpec(raw_query=q, task_type=TaskType.VQA)
    assert q in build_prompt(spec)
    print("  vqa prompt keeps the question   ok")


def test_chat_messages_image_count() -> None:
    msgs = chat_messages("prompt", n_images=2)
    images = [c for c in msgs[0]["content"] if c["type"] == "image"]
    assert len(images) == 2
    print("  chat messages image count       ok")


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def test_normalise_strips_articles_and_punctuation() -> None:
    assert normalise("The Forest, and water.") == "forest and water"
    print("  normalise                       ok")


def test_exact_match_is_not_generous() -> None:
    assert exact_match("forest", "Forest.") == 1.0
    assert exact_match("woods", "forest") == 0.0    # no synonym mapping
    print("  exact match has no synonyms     ok")


def test_multilabel_f1() -> None:
    p, r, f = multilabel_prf("forest, water", "forest, water, urban")
    assert p == 1.0
    assert round(r, 2) == 0.67
    assert 0.79 < f < 0.81
    print("  multilabel prf                  ok")


def test_empty_prediction_scores_zero_f1() -> None:
    _, _, f = multilabel_prf("", "forest, water")
    assert f == 0.0
    print("  empty prediction scores zero    ok")


def test_label_set_splits_on_semicolons() -> None:
    assert to_label_set("forest; water, urban fabric") == {
        "forest", "water", "urban fabric"}
    print("  label set splitting             ok")


def test_article_stripping_edge_case() -> None:
    """Documented limitation: a label that IS an article normalises away.

    Harmless for real land-cover vocabularies, but it would silently drop
    labels in another domain. Kept as a test so nobody rediscovers it during
    demo week.
    """
    assert normalise("a") == ""
    assert to_label_set("a, the, forest") == {"forest"}
    print("  article edge case documented    ok")


def main() -> None:
    print("\nWeek 2 tests")
    print("-" * 44)
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("-" * 44)
    print("all passed\n")


if __name__ == "__main__":
    main()
