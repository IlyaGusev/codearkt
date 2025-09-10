FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml README.md /app/
COPY codearkt /app/codearkt
RUN pip install --no-cache-dir .
COPY examples /app/examples

EXPOSE 5055
ENV PHOENIX_URL=http://phoenix:6006 \
    PHOENIX_PROJECT_NAME=codearkt \
    ACADEMIA_MCP_URL=http://academia:5056/mcp \
    CODEARKT_EXECUTOR_URL=http://executor:8000

CMD ["python", "-m", "examples.multi_agent.run", "--port", "5055"]

