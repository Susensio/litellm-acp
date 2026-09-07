default: check

# Install the locked development environment.
sync:
    uv sync --all-groups

lint:
    uv run ruff check .

format:
    uv run ruff format --check .

test:
    uv run pytest

typecheck:
    uv run ty check src

# Run every currently supported, non-mutating project check.
check: lint format typecheck test
