# financial-advisor

Multi-agent financial-advisor system built with LangGraph. It orchestrates a
supervisor plus specialist workers (market data, news, optimizer, and RAG) to
answer portfolio queries end-to-end.

> **Prerequisites**: set the required API keys in `.env` (copy from
> `.env.example`), notably `OPENAI_API_KEY` (embeddings) and
> `DEEPSEEK_API_KEY` (supervisor LLM).

## Graph Setup & Execution

### Step 1: Vector Database Initialization

Initialize ChromaDB and ingest the PDFs in `rag_files/` into the vector store:

```bash
python init_chroma.py
```

This scans `rag_files/` for PDFs, chunks each document (500-character chunks with
50-character overlap), embeds every chunk, and stores the vectors with metadata
in a local persistent ChromaDB at `./chroma_db`.

### Step 2: Local Execution via LangGraph

Launch the application locally using the LangGraph development environment:

```bash
langgraph dev
```

Then open the Studio UI at the URL printed in the terminal to interact with the
graph and inspect its execution.

### Key Technical Information Notes

- **Supervisor Node**: Powered by **DeepSeek v4 Pro** for dynamic task planning
  and intent routing.
- **RAG & Embeddings**: ChromaDB utilizes the **OpenAI API**
  (`text-embedding-3-small`) for document embeddings and hybrid search.

### Sample Test Payloads

Paste any of these JSON objects into LangGraph Studio (`langgraph dev`) or an API
call. Each payload combines the test portfolio, the constraints, and one of the
five test queries.

**Query 1 — RAG / Historical Analysis**

```json
{
  "query": "How has AAPL's financial performance been over the last 5 years?",
  "portfolio": [
    { "ticker": "AAPL", "shares": 100 },
    { "ticker": "MSFT", "shares": 50 },
    { "ticker": "GOOGL", "shares": 25 }
  ],
  "constraints": {
    "maxRisk": "medium",
    "minDividendYield": 2.0,
    "sector": "technology"
  }
}
```

**Query 2 — RAG / Risk Factors**

```json
{
  "query": "What are MSFT's risk factors?",
  "portfolio": [
    { "ticker": "AAPL", "shares": 100 },
    { "ticker": "MSFT", "shares": 50 },
    { "ticker": "GOOGL", "shares": 25 }
  ],
  "constraints": {
    "maxRisk": "medium",
    "minDividendYield": 2.0,
    "sector": "technology"
  }
}
```

**Query 3 — RAG / Industry Trends**

```json
{
  "query": "What trends are being seen in the Tech sector?",
  "portfolio": [
    { "ticker": "AAPL", "shares": 100 },
    { "ticker": "MSFT", "shares": 50 },
    { "ticker": "GOOGL", "shares": 25 }
  ],
  "constraints": {
    "maxRisk": "medium",
    "minDividendYield": 2.0,
    "sector": "technology"
  }
}
```

**Query 4 — Optimizer / Rebalancing**

```json
{
  "query": "Optimize this portfolio",
  "portfolio": [
    { "ticker": "AAPL", "shares": 100 },
    { "ticker": "MSFT", "shares": 50 },
    { "ticker": "GOOGL", "shares": 25 }
  ],
  "constraints": {
    "maxRisk": "medium",
    "minDividendYield": 2.0,
    "sector": "technology"
  }
}
```

**Query 5 — News / Sentiment**

```json
{
  "query": "What do the latest news say about the portfolio?",
  "portfolio": [
    { "ticker": "AAPL", "shares": 100 },
    { "ticker": "MSFT", "shares": 50 },
    { "ticker": "GOOGL", "shares": 25 }
  ],
  "constraints": {
    "maxRisk": "medium",
    "minDividendYield": 2.0,
    "sector": "technology"
  }
}
```

## Backend Setup & Local Execution

### Step 1: Start the Python LangGraph Development Server

The Python graph server must be started first to serve the multi-agent system:

```bash
langgraph dev
```

*(Runs locally on port `2024`)*

### Step 2: Start the Express.js Backend

Navigate into the `backend` directory, install dependencies (if not already
done), and launch the server:

```bash
cd backend
npm install
npm start
```

*(Runs locally on port `3000`)*

### Step 3: Access Swagger API Documentation

Once the Express backend is running, access the interactive Swagger OpenAPI
documentation at:

- **URL**: `http://localhost:3000/api/docs`

## Docker Deployment

Run the entire system (Python LangGraph server + Express.js backend + persistent
ChromaDB volume) with Docker Compose.

### Prerequisites & Environment Setup

A valid `.env` file containing the required API keys (e.g., `OPENAI_API_KEY`,
`DEEPSEEK_API_KEY`) must exist at the project root.

### Step-by-Step Execution Commands

Build and start the containers in the background:

```bash
docker compose up --build -d
```

View running container logs:

```bash
docker compose logs -f
```

Stop and tear down the containers:

```bash
docker compose down
```

### ChromaDB Volume & Ingestion

The compose file bind-mounts the host's ChromaDB directory into the graph
container (`./chroma_db:/app/chroma_db`), so both share the same vector-store
files:

- **Already ingested locally?** No action needed — the existing `./chroma_db`
  is used as-is inside the container.
- **Fresh clone / empty `./chroma_db`?** Populate it once before querying RAG:

  ```bash
  docker compose run --rm python-graph python init_chroma.py
  ```

- **Added or changed PDFs in `rag_files/`?** Re-run the same command to
  re-ingest.

This one-time ingestion writes to the same `./chroma_db` used by the running
graph server (whether run on the host or inside the container), so it only needs
to run when the vector store is empty or the documents change.

### Service Access Information

- **Express Backend & Swagger Docs**: `http://localhost:3000/api/docs`
- **Health Endpoint**: `http://localhost:3000/api/health`
- **Python LangGraph Server**: `http://localhost:2024`


