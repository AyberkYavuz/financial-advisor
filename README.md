# financial-advisor

Multi-agent financial-advisor system built with LangGraph. It orchestrates a
supervisor plus specialist workers (market data, news, optimizer, and RAG) to
answer portfolio queries end-to-end.

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

> **Prerequisites**: set the required API keys in `.env` (copy from
> `.env.example`), notably `OPENAI_API_KEY` (embeddings) and
> `DEEPSEEK_API_KEY` (supervisor LLM).
