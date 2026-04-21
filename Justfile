default:
    just --list

sync-api:
    uv sync --extra api

sync-front:
    uv sync --extra front

sync-notebooks:
    uv sync --extra notebooks

sync-dev:
    uv sync --extra dev

sync-all:
    uv sync --extra api --extra front --extra notebooks --extra dev

# Dev local
api:
    uv run uvicorn api.main:app --reload --host 127.0.0.1 --port 8000

front:
    uv run streamlit run front/app.py

dev:
    just api & just front

lint:
    uv run ruff check .

test:
    uv run pytest