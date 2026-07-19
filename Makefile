.PHONY: contract test lint verify skill

contract:
	python scripts/verify_model_contract.py

test:
	pytest -q

lint:
	python -m ruff check .
	python -m ruff format --check src tests scripts benchmarks

skill:
	python skills/writing-cute-dsl-kernels/scripts/validate_skill.py
	cd skills/writing-cute-dsl-kernels && sha256sum -c SHA256SUMS.txt

verify:
	bash scripts/verify_bundle.sh
