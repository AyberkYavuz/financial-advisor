FROM python:3.12-slim

WORKDIR /app

# System dependencies for building native extensions (numpy/pandas/etc.).
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv (required by `langgraph dev`).
RUN pip install --no-cache-dir uv

# Install the LangGraph CLI (provides the `langgraph` command).
RUN pip install --no-cache-dir "langgraph-cli[inmem]"

# Install Python dependencies.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy project source.
COPY pyproject.toml ./
COPY graph.py ./
COPY graph ./graph
COPY init_chroma.py ./
COPY langgraph.json ./
COPY rag_files ./rag_files

EXPOSE 2024

CMD ["langgraph", "dev", "--host", "0.0.0.0", "--port", "2024", "--no-browser", "--no-reload"]
