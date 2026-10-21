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

/**
 * @swagger
 * /api/health:
 *   get:
 *     summary: Health check
 *     description: Returns the server health status.
 *     tags: [Health]
 *     responses:
 *       200:
 *         description: Server is healthy.
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 status:
 *                   type: string
 *                   example: healthy
 *                 service:
 *                   type: string
 *                   example: forinvest-backend
 */
router.get("/health", (req, res) => {
  res.status(200).json({ status: "healthy", service: "forinvest-backend" });
});

/**
 * @swagger
 * /api/invoke:
 *   post:
 *     summary: Invoke the LangGraph system
 *     description: >-
 *       Runs the multi-agent graph with a user query, portfolio, and
 *       constraints, returning the final advisory result.
 *     tags: [Invoke]
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             required:
 *               - query
 *             properties:
 *               query:
 *                 type: string
 *                 description: The user query to run through the graph.
 *                 example: Optimize this portfolio
 *               portfolio:
 *                 type: array
 *                 description: List of holdings.
 *                 items:
 *                   type: object
 *                   properties:
 *                     ticker:
 *                       type: string
 *                       example: AAPL
 *                     shares:
 *                       type: integer
 *                       example: 100
 *               constraints:
 *                 type: object
 *                 description: Investment constraints.
 *                 properties:
 *                   maxRisk:
 *                     type: string
 *                     example: medium
 *                   minDividendYield:
 *                     type: number
 *                     example: 2.0
 *                   sector:
 *                     type: string
 *                     example: technology
 *     responses:
 *       200:
 *         description: Graph execution result.
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 thread_id:
 *                   type: string
 *                 result:
 *                   type: object
 *                   description: Final graph state (advisory report).
 *       400:
 *         description: Bad request (missing or invalid query).
 *       500:
 *         description: Internal server error (graph invocation failed).
 */
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
