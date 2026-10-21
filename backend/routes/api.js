const express = require("express");
const { Client } = require("@langchain/langgraph-sdk");

const router = express.Router();

// Point at the local LangGraph dev server (or override via env).
const LANGGRAPH_API_URL =
  process.env.LANGGRAPH_API_URL || "http://127.0.0.1:2024";
const GRAPH_ID = process.env.GRAPH_ID || "agent";

// Lazily-initialized singleton client, reused across requests.
let client = null;
function getClient() {
  if (!client) {
    client = new Client({ apiUrl: LANGGRAPH_API_URL });
  }
  return client;
}

// GET /api/health — liveness/readiness probe.
router.get("/health", (req, res) => {
  res.status(200).json({ status: "healthy", service: "forinvest-backend" });
});

// POST /api/invoke — run the graph with a query + portfolio + constraints.
router.post("/invoke", async (req, res) => {
  try {
    const { query, portfolio = [], constraints = {} } = req.body || {};

    if (!query || typeof query !== "string" || !query.trim()) {
      return res.status(400).json({ error: "query is required" });
    }

    const input = { query, portfolio, constraints };
    const client = getClient();

    // Create a thread, then run the graph and wait for the final state.
    const thread = await client.threads.create();
    const result = await client.runs.wait(thread.thread_id, GRAPH_ID, {
      input,
    });

    return res.status(200).json({ thread_id: thread.thread_id, result });
  } catch (error) {
    return res.status(500).json({
      error: error && error.message ? error.message : "Internal server error",
    });
  }
});

module.exports = router;
