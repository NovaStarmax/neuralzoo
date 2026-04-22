FROM python:3.13-slim

WORKDIR /app

RUN pip install uv

COPY pyproject.toml uv.lock ./
RUN uv sync --extra front --no-dev --no-install-project

COPY front/ front/

ENV STREAMLIT_SERVER_HEADLESS=true

CMD ["uv", "run", "streamlit", "run", "front/app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
