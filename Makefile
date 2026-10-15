.PHONY: pipeline demo predict predict-file preprocess split train-baseline train-improved error-analysis test lint format fix typecheck check

# The whole flow from the raw CSV to the saved model and its analysis
pipeline: preprocess split train-baseline train-improved error-analysis

demo:
	uv run python -m itg_classification.demo

# make predict TEXT="Profit rose 20% year-on-year"
predict:
	uv run itg-classification "$(TEXT)"

# make predict-file INPUT=news.csv OUTPUT=predictions.csv
predict-file:
	uv run itg-classification --input "$(INPUT)" --output "$(OUTPUT)"

preprocess:
	uv run python -m itg_classification.preprocess

split:
	uv run python -m itg_classification.split

train-baseline:
	uv run python -m itg_classification.train_baseline

train-improved:
	uv run python -m itg_classification.train_improved

error-analysis:
	uv run python -m itg_classification.error_analysis

test:
	uv run pytest

lint:
	uv run ruff check .

format:
	uv run ruff format .

fix:
	uv run ruff check --fix .
	uv run ruff format .

typecheck:
	uv run ty check

check:
	@./scripts/check.sh
