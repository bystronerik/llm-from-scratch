# llm demo

## Linting

This project uses [ruff](https://docs.astral.sh/ruff/) (managed via [uv](https://docs.astral.sh/uv/)).

```bash
uv run ruff check .        # lint
uv run ruff check --fix .  # lint and autofix
uv run ruff format .       # format
```

## Pre-commit hook

Linting and formatting run automatically before each commit via
[pre-commit](https://pre-commit.com/). After cloning, install the git hook once:

```bash
uv run pre-commit install
```

Ruff then runs on staged files at commit time and blocks the commit on failures.

