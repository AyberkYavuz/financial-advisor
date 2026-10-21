const express = require("express");
const apiRouter = require("./routes/api");

const app = express();

// Parse JSON request bodies.
app.use(express.json());

// Mount the API routes under /api.
app.use("/api", apiRouter);

const PORT = process.env.PORT || 3000;

// Start listening only when run directly (not when imported by tests).
if (require.main === module) {
  app.listen(PORT, () => {
    // eslint-disable-next-line no-console
    console.log(`forinvest-backend listening on http://localhost:${PORT}`);
  });
}

module.exports = app;
