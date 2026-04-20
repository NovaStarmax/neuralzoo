FROM python:3.13-slim

WORKDIR /app

RUN pip install uv

COPY pyproject.toml .
COPY uv.lock .
RUN uv sync --extra front

COPY front/ front/

CMD ["uv", "run", "streamlit", "run", "front/app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
