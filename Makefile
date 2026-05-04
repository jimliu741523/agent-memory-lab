.PHONY: test bench bench-multi demo clean

PYTHON ?= python3

test:
	$(PYTHON) -m unittest discover tests -v

bench:
	$(PYTHON) -m bench.run

bench-multi:
	$(PYTHON) -m bench.run --multi-seed 10

bench-save:
	$(PYTHON) -m bench.run --output bench/results/results.md

demo:
	@echo "=== sliding_window ==="
	$(PYTHON) -m patterns.sliding_window
	@echo
	@echo "=== summary_compression ==="
	$(PYTHON) -m patterns.summary_compression
	@echo
	@echo "=== hierarchical_summary ==="
	$(PYTHON) -m patterns.hierarchical_summary
	@echo
	@echo "=== vector_retrieval ==="
	$(PYTHON) -m patterns.vector_retrieval
	@echo
	@echo "=== structured_episodic ==="
	$(PYTHON) -m patterns.structured_episodic

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
