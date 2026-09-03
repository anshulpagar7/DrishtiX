# SatQuery AI - common tasks
#
#   make install    runtime deps only (no GPU needed)
#   make run        launch the app
#   make test       all three test suites
#   make eval       both benchmarks
#   make demo       rehearse the demo + write deck assets
#   make api        run the backend + site on :8000
#   make check      everything CI runs, locally

PY ?= python3

.PHONY: help install install-train run test eval demo assets check clean

help:
	@grep -E '^#   ' Makefile | sed 's/^#   //'

install:
	$(PY) -m pip install -r requirements.txt

install-train:          ## GPU machines only (Kaggle / Colab)
	$(PY) -m pip install -r requirements-train.txt

run:
	$(PY) -m streamlit run app.py

api:                    ## backend + multi-page site on http://localhost:8000
	$(PY) -m uvicorn api.main:app --reload --port 8000

install-api:
	$(PY) -m pip install -r requirements-api.txt

test:
	$(PY) -m tests.test_contract
	$(PY) -m tests.test_week2
	$(PY) -m tests.test_week3
	$(PY) -m tests.test_week4

eval:
	$(PY) -m eval.run_eval
	$(PY) -m eval.change_eval

demo:
	$(PY) -m demo.script --no-images

assets:                 ## write deck-ready PNGs to demo/output
	$(PY) -m demo.script

check: test eval demo
	$(PY) -m compileall -q .
	@echo "\nAll checks passed."

clean:
	rm -rf .cache .data .checkpoints eval/reports demo/output demo/scenes
	find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
	@echo "Cleaned. Note: .checkpoints removed - re-download adapters if needed."
