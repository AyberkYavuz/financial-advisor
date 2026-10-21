const request = require("supertest");

// Mock the LangGraph SDK client so tests run without a live graph server.
const mockThreadsCreate = jest.fn().mockResolvedValue({ thread_id: "thread-123" });
const mockRunsWait = jest.fn().mockResolvedValue({
  final_response: "advisory report",
  next_node: "FINISH",
});

jest.mock("@langchain/langgraph-sdk", () => ({
  Client: jest.fn().mockImplementation(() => ({
    threads: { create: mockThreadsCreate },
    runs: { wait: mockRunsWait },
  })),
}));

const app = require("../server");

describe("GET /api/health", () => {
  test("returns 200 OK with the health payload", async () => {
    const res = await request(app).get("/api/health");

    expect(res.status).toBe(200);
    expect(res.body).toEqual({ status: "healthy", service: "forinvest-backend" });
  });
});

describe("POST /api/invoke", () => {
  test("returns 400 when the query is missing", async () => {
    const res = await request(app).post("/api/invoke").send({});

    expect(res.status).toBe(400);
    expect(res.body).toEqual({ error: "query is required" });
  });

  test("invokes the graph and returns the result", async () => {
    const res = await request(app)
      .post("/api/invoke")
      .send({
        query: "Optimize this portfolio",
        portfolio: [
          { ticker: "AAPL", shares: 100 },
          { ticker: "MSFT", shares: 50 },
          { ticker: "GOOGL", shares: 25 },
        ],
        constraints: {
          maxRisk: "medium",
          minDividendYield: 2.0,
          sector: "technology",
        },
      });

    expect(res.status).toBe(200);
    expect(res.body.thread_id).toBe("thread-123");
    expect(res.body.result).toBeDefined();
    expect(res.body.result.final_response).toBe("advisory report");

    // The SDK was called with the graph input.
    expect(mockThreadsCreate).toHaveBeenCalled();
    expect(mockRunsWait).toHaveBeenCalledWith(
      "thread-123",
      "agent",
      expect.objectContaining({
        input: {
          query: "Optimize this portfolio",
          portfolio: [
            { ticker: "AAPL", shares: 100 },
            { ticker: "MSFT", shares: 50 },
            { ticker: "GOOGL", shares: 25 },
          ],
          constraints: {
            maxRisk: "medium",
            minDividendYield: 2.0,
            sector: "technology",
          },
        },
      })
    );
  });

  test("returns 500 when the graph invocation fails", async () => {
    mockRunsWait.mockRejectedValueOnce(new Error("graph unavailable"));

    const res = await request(app)
      .post("/api/invoke")
      .send({ query: "Hello" });

    expect(res.status).toBe(500);
    expect(res.body).toEqual({ error: "graph unavailable" });
  });
});
