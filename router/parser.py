"""
Query understanding: natural language -> TaskSpec.

Week 1 ships the rule-based parser. It runs offline, instantly, for free, and
gives you a measurable baseline. Week 2+ you add LLMParser behind the same
Parser interface and report the accuracy delta - that comparison is a slide.

Do not delete the rule parser when the LLM parser lands. It is your fallback
when a free-tier API rate-limits you mid-demo.
"""

from __future__ import annotations

import json
import re
from typing import Any, Protocol

from contracts import Modality, TaskSpec, TaskType


class Parser(Protocol):
    """Anything that turns a query into a TaskSpec."""

    parser_id: str

    def parse(self, query: str, n_images: int = 1) -> TaskSpec: ...


# --------------------------------------------------------------------------
# Keyword tables. Order matters: CHANGE is checked before everything else
# because "what changed" also contains question words.
# --------------------------------------------------------------------------

_CHANGE_PATTERNS = [
    r"\bchang(e|ed|es|ing)\b", r"\bdiffer(ence|ent)\b", r"\bbefore\b.*\bafter\b",
    r"\bcompare\b", r"\bsince\b", r"\bgrown?\b", r"\bshrunk\b", r"\bnew\b.*\bbuilt\b",
    r"\bdeforest", r"\bexpansion\b", r"\bbetween\s+\d{4}\s+and\s+\d{4}",
]

_GROUND_PATTERNS = [
    r"\bwhere\s+(is|are)\b", r"\blocate\b", r"\bfind\s+the\b", r"\bpoint\s+(out|to)\b",
    r"\bshow\s+me\s+the\b", r"\bwhich\s+part\b",
]

_CAPTION_PATTERNS = [
    r"\bdescribe\b", r"\bwhat.s\s+in\s+th(is|e)\b", r"\bsummari[sz]e\b",
    r"\bcaption\b", r"\btell\s+me\s+about\b", r"\boverview\b",
]

_CLASSIFY_PATTERNS = [
    r"\bland\s*cover\b", r"\bland\s*use\b", r"\bclassif", r"\bwhat\s+type\s+of\s+terrain\b",
    r"\bcategor", r"\bwhat\s+kind\s+of\s+(area|region|land)\b",
]

# Land-cover vocabulary, loosely BigEarthNet-flavoured. Extend as you go.
_CLASS_VOCAB = [
    "water", "river", "lake", "sea", "ocean", "wetland", "marsh",
    "forest", "trees", "woodland", "vegetation", "grassland", "pasture",
    "farmland", "cropland", "agriculture", "field", "orchard", "vineyard",
    "urban", "city", "buildings", "settlement", "residential", "industrial",
    "road", "highway", "runway", "airport", "port", "harbour", "harbor",
    "bare soil", "sand", "beach", "desert", "rock", "snow", "ice", "glacier",
    "burnt", "fire scar", "flood", "flooded",
]

_SAR_HINTS = [r"\bsar\b", r"\bsentinel[\s-]?1\b", r"\bradar\b", r"\bbackscatter\b",
              r"\bvv\b", r"\bvh\b", r"\bthrough\s+cloud"]
_OPTICAL_HINTS = [r"\boptical\b", r"\bsentinel[\s-]?2\b", r"\brgb\b", r"\btrue\s+colou?r\b",
                  r"\bmultispectral\b"]

_TEMPORAL_HINTS = [r"\b(19|20)\d{2}\b", r"\blast\s+(year|month|decade)\b",
                   r"\bover\s+time\b", r"\btime\s+series\b", r"\bhistoric"]


def _any(patterns: list[str], text: str) -> bool:
    return any(re.search(p, text) for p in patterns)


def _extract_class(text: str) -> str | None:
    """Longest matching vocabulary term wins, so 'bare soil' beats 'soil'."""
    hits = [c for c in _CLASS_VOCAB if re.search(rf"\b{re.escape(c)}\b", text)]
    return max(hits, key=len) if hits else None


class RuleParser:
    """Deterministic keyword parser. No network, no model, no cost."""

    parser_id = "rule-v1"

    def parse(self, query: str, n_images: int = 1) -> TaskSpec:
        text = query.lower().strip()

        # --- task type ----------------------------------------------------
        if _any(_CHANGE_PATTERNS, text):
            task, conf = TaskType.CHANGE, 0.85
        elif _any(_GROUND_PATTERNS, text):
            task, conf = TaskType.GROUND, 0.8
        elif _any(_CLASSIFY_PATTERNS, text):
            task, conf = TaskType.CLASSIFY, 0.8
        elif _any(_CAPTION_PATTERNS, text):
            task, conf = TaskType.CAPTION, 0.8
        elif text.endswith("?") or re.match(r"^(what|how|is|are|does|do|can|why|when)\b", text):
            task, conf = TaskType.VQA, 0.6
        else:
            task, conf = TaskType.UNKNOWN, 0.2

        # --- modality -----------------------------------------------------
        if _any(_SAR_HINTS, text):
            modality = Modality.SAR
        elif _any(_OPTICAL_HINTS, text):
            modality = Modality.OPTICAL
        else:
            modality = Modality.ANY

        # --- flags --------------------------------------------------------
        needs_pair = task is TaskType.CHANGE
        temporal = needs_pair or _any(_TEMPORAL_HINTS, text)

        return TaskSpec(
            raw_query=query,
            task_type=task,
            modality=modality,
            needs_pair=needs_pair,
            temporal=temporal,
            target_class=_extract_class(text),
            confidence=conf,
            parser_id=self.parser_id,
        )


class LLMParser:
    """Week 2. Free-tier chat API, strict JSON out, cached, fallback-safe.

    Three guarantees, in priority order:
      1. Never raises. A failed parse falls back to RuleParser.
      2. Never repeats a call. Every completion is cached to disk.
      3. Never required. With SATQUERY_LLM_BACKEND unset the backend is
         offline and this class transparently becomes RuleParser.

    That third property is why you can hand this to a teammate without them
    needing an API key to run the app.
    """

    parser_id = "llm-v1"

    SYSTEM = """You classify questions asked about satellite imagery.

Task types:
  classify - which land cover classes are present
  caption  - describe the whole scene
  ground   - locate a specific named thing in the image
  change   - compare two images of the same place at different times
  vqa      - any other specific question about one image

Rules:
  - Anything comparing across time is "change", even when phrased as a
    question ("has the glacier retreated?", "2019 vs 2023", "is the lake
    smaller now?").
  - needs_pair is true if and only if task_type is "change".
  - target_class is the land cover noun being asked about, or null.

Return ONLY a JSON object. No prose, no markdown fences.
{"task_type":"classify|vqa|caption|ground|change",
 "modality":"optical|sar|any",
 "needs_pair":true|false,
 "temporal":true|false,
 "target_class":"string or null",
 "confidence":0.0-1.0}

Question: """

    def __init__(self, backend: Any = None, fallback: Parser | None = None,
                 cache: Any = None, use_cache: bool = True) -> None:
        from router.llm_backends import ParseCache, get_backend

        self.backend = backend if backend is not None else get_backend()
        self.fallback = fallback or RuleParser()
        self.cache = cache if cache is not None else (ParseCache() if use_cache else None)
        self.last_source = "unset"      # "cache" | "llm" | "fallback"

    # ----------------------------------------------------------------------

    def _completion(self, query: str) -> str | None:
        prompt = self.SYSTEM + query
        ck = None
        if self.cache is not None:
            from router.llm_backends import ParseCache

            ck = ParseCache.key(self.backend.backend_id, self.backend.model, prompt)
            hit = self.cache.get(ck)
            if hit is not None:
                self.last_source = "cache"
                return hit

        raw = self.backend.complete(prompt)
        if raw is not None:
            self.last_source = "llm"
            if self.cache is not None and ck:
                self.cache.put(ck, raw)
        return raw

    @staticmethod
    def _extract_json(raw: str) -> dict:
        """Models add fences and prose no matter how firmly you ask."""
        text = raw.replace("```json", "").replace("```", "").strip()
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end == -1:
            raise ValueError("no JSON object in completion")
        return json.loads(text[start:end + 1])

    # ----------------------------------------------------------------------

    def parse(self, query: str, n_images: int = 1) -> TaskSpec:
        try:
            raw = self._completion(query)
            if raw is None:
                raise ValueError("backend unavailable")

            d = self._extract_json(raw)
            task = TaskType(str(d["task_type"]).lower())
            spec = TaskSpec(
                raw_query=query,
                task_type=task,
                modality=Modality(str(d.get("modality", "any")).lower()),
                # Trust the contract over the model: only change needs a pair.
                needs_pair=(task is TaskType.CHANGE),
                temporal=bool(d.get("temporal", task is TaskType.CHANGE)),
                target_class=(d.get("target_class") or None),
                confidence=float(d.get("confidence", 0.7)),
                parser_id=self.parser_id,
            )
            return spec
        except Exception:
            # Never let the demo die because an API rate-limited us.
            self.last_source = "fallback"
            return self.fallback.parse(query, n_images)


def build_parser(prefer_llm: bool = True) -> Parser:
    """What the app and eval harness both call.

    Returns the LLM parser when a backend is actually configured, otherwise
    the rule parser. No caller needs to know which it got.
    """
    if not prefer_llm:
        return RuleParser()
    from router.llm_backends import get_backend

    backend = get_backend()
    return LLMParser(backend=backend) if backend.available() else RuleParser()
