const express = require("express");
const morgan = require("morgan");
const swaggerUi = require("swagger-ui-express");
const swaggerSpec = require("./swagger");
const apiRouter = require("./routes/api");

const app = express();

// HTTP request logging: timestamp, method, URL, status, response time.
app.use(morgan(":date[iso] :method :url :status :response-time ms"));

// Parse JSON request bodies.
app.use(express.json());

// Swagger UI + OpenAPI docs.
app.use("/api/docs", swaggerUi.serve, swaggerUi.setup(swaggerSpec));

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
