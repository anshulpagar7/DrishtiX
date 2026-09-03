"""
HTTP API for DrishtiX.

Thin on purpose. All the reasoning lives in `pipeline.py`; this layer only
decodes images, calls `ask()`, and serialises the Answer. If you find yourself
writing logic here, it belongs in the pipeline where it can be unit-tested
without a web server.

Run:
    pip install -r requirements-api.txt
    uvicorn api.main:app --reload --port 8000

Then open http://localhost:8000 — the site is served from the same origin, so
there is no CORS to configure for the normal case.

Endpoints:
    GET  /api/health      what is loaded, and what is not
    GET  /api/models      the registry, as the router sees it
    GET  /api/scenes      built-in synthetic scenes (no upload needed)
    POST /api/ask         image(s) + question -> grounded answer
    POST /api/ask/scene   same, against a built-in scene
"""

from __future__ import annotations

import base64
import io
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, File, Form, HTTPException, UploadFile  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware                  # noqa: E402
from fastapi.responses import FileResponse                          # noqa: E402
from fastapi.staticfiles import StaticFiles                         # noqa: E402
from PIL import Image                                               # noqa: E402

import config                                                       # noqa: E402
from api.schemas import (AnswerOut, EvidenceOut, HealthOut,         # noqa: E402
                         ModelOut, OutputOut, PlanOut, SceneOut)
from contracts import Answer                                        # noqa: E402
from demo.samples import SCENES, build as build_scene               # noqa: E402
from pipeline import build_pipeline                                 # noqa: E402
from render import render_answer                                    # noqa: E402

MAX_UPLOAD_BYTES = 12 * 1024 * 1024
MAX_IMAGES = 2

app = FastAPI(
    title="DrishtiX API",
    version="0.4.0",
    description="Agentic vision-language assistant for satellite imagery.",
)

# Same-origin in normal use. Kept open for the case where the front end is
# served separately during development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Built once at import. Loading weights per request would make every call slow
# and would race under concurrency.
_pipeline = build_pipeline(with_vlm=True)


@app.on_event("startup")
def _warm() -> None:
    """Pre-build the scene payload. Doing it lazily meant the first visitor
    waited on it while the console sat on 'Checking backend…'."""
    scenes()


# --------------------------------------------------------------------------
# Serialisation
# --------------------------------------------------------------------------

def _png_data_uri(image: Image.Image) -> str:
    """PNG data URI. No optimize= — it costs ~0.5s per noisy tile and saves
    a few percent of bytes, which is a bad trade for a live console."""
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG", compress_level=1)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _serialise(answer: Answer, images: list[Image.Image],
               overlays: bool = True) -> AnswerOut:
    plan = None
    if answer.plan:
        plan = PlanOut(
            models=answer.plan.model_ids,
            fusion=answer.plan.fusion.value,
            rationale=answer.plan.rationale,
        )

    rendered: dict[str, str] = {}
    if overlays:
        for caption, img in render_answer(images, answer).items():
            rendered[caption] = _png_data_uri(img)

    return AnswerOut(
        answered=answer.answered,
        text=answer.text,
        task=answer.spec.task_type.value if answer.spec else "unknown",
        modality=answer.spec.modality.value if answer.spec else "any",
        needs_pair=bool(answer.spec.needs_pair) if answer.spec else False,
        parser=answer.spec.parser_id if answer.spec else "unknown",
        parser_confidence=answer.spec.confidence if answer.spec else 0.0,
        refusal_reason=(answer.validation.reason
                        if answer.validation and not answer.validation.ok else None),
        fix_hint=(answer.validation.fix_hint
                  if answer.validation and not answer.validation.ok else None),
        plan=plan,
        route=answer.route,
        outputs=[
            OutputOut(
                model_id=o.model_id,
                answer=o.answer,
                confidence=o.confidence,
                latency_ms=round(o.latency_ms, 1),
                veto=o.veto,
                scores=o.scores,
            )
            for o in answer.outputs
        ],
        evidence=[
            EvidenceOut(
                image_index=e.image_index,
                kind=e.kind.value,
                bbox=list(e.bbox) if e.bbox else None,
                score=e.score,
                note=e.note,
            )
            for e in answer.evidence
        ],
        overlays=rendered,
        latency_ms=round(answer.total_latency_ms, 1),
    )


async def _read_images(files: list[UploadFile]) -> list[Image.Image]:
    if len(files) > MAX_IMAGES:
        raise HTTPException(400, f"At most {MAX_IMAGES} images per request.")

    images: list[Image.Image] = []
    for f in files:
        raw = await f.read()
        if not raw:
            continue
        if len(raw) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                413, f"{f.filename} exceeds {MAX_UPLOAD_BYTES // (1024*1024)} MB.")
        try:
            images.append(Image.open(io.BytesIO(raw)).convert("RGB"))
        except Exception:
            raise HTTPException(
                400, f"{f.filename} is not a readable image.")
    return images


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------

@app.get("/api/health", response_model=HealthOut)
def health() -> HealthOut:
    """What is actually loaded. Deliberately blunt — a demo that silently
    falls back to baselines while the deck claims a fine-tuned VLM is how
    teams get caught in Q&A."""
    from models.rsvqa import RSVQAModel

    probe = RSVQAModel()
    probe.ensure_loaded()      # cheap: returns False immediately if absent
    registry = _pipeline.registry.describe()

    return HealthOut(
        status="ok",
        parser=getattr(_pipeline.parser, "parser_id", "unknown"),
        llm_backend=config.LLM_BACKEND,
        rsvqa_adapter_present=probe.adapter_present,
        rsvqa_status=probe.status,
        models_loaded=[m["model_id"] for m in registry if m["loaded"]],
        models_registered=[m["model_id"] for m in registry],
        scenes=list(SCENES),
    )


@app.get("/api/models", response_model=list[ModelOut])
def models() -> list[ModelOut]:
    return [
        ModelOut(model_id=m["model_id"], tasks=m["tasks"],
                 priority=m["priority"], loaded=m["loaded"])
        for m in _pipeline.registry.describe()
    ]


@app.get("/api/scenes", response_model=list[SceneOut])
def scenes() -> list[SceneOut]:
    """Built-in synthetic scenes, so the console works with no upload and no
    network. This is the offline demo path."""
    out: list[SceneOut] = []
    for name in SCENES:
        before, after, _, desc = build_scene(name, 384)
        out.append(SceneOut(
            name=name,
            description=desc,
            before=_png_data_uri(before),
            after=_png_data_uri(after),
        ))
    return out


@app.post("/api/ask", response_model=AnswerOut)
async def ask(
    query: str = Form(...),
    images: list[UploadFile] = File(default=[]),
    overlays: bool = Form(default=True),
) -> AnswerOut:
    if not query.strip():
        raise HTTPException(400, "query must not be empty.")

    pil = await _read_images(images)
    answer = _pipeline.ask(query, pil)
    return _serialise(answer, pil, overlays)


@app.post("/api/ask/scene", response_model=AnswerOut)
async def ask_scene(
    query: str = Form(...),
    scene: str = Form(...),
    pair: bool = Form(default=True),
    overlays: bool = Form(default=True),
) -> AnswerOut:
    # Same guard as /api/ask. The pipeline degrades gracefully on an empty
    # query, but the two endpoints disagreeing about what is a 400 is the
    # kind of inconsistency that bites whoever writes the client.
    if not query.strip():
        raise HTTPException(400, "query must not be empty.")
    if scene not in SCENES:
        raise HTTPException(404, f"Unknown scene '{scene}'. Known: {list(SCENES)}")

    before, after, _, _ = build_scene(scene, 384)
    pil = [before, after] if pair else [before]
    answer = _pipeline.ask(query, pil)
    return _serialise(answer, pil, overlays)


# --------------------------------------------------------------------------
# Static site, mounted last so /api/* wins
# --------------------------------------------------------------------------

SITE = ROOT / "site"
if SITE.is_dir():
    app.mount("/assets", StaticFiles(directory=SITE / "assets"), name="assets")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(SITE / "index.html")

    @app.get("/{page}.html", include_in_schema=False)
    def page(page: str) -> FileResponse:
        target = (SITE / f"{page}.html").resolve()
        # Path containment check: never serve outside site/.
        if not str(target).startswith(str(SITE.resolve())) or not target.is_file():
            raise HTTPException(404, "No such page.")
        return FileResponse(target)
