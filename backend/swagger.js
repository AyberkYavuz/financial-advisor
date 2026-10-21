const path = require("path");
const swaggerJsdoc = require("swagger-jsdoc");

const PORT = process.env.PORT || 3000;

const options = {
  definition: {
    openapi: "3.0.0",
    info: {
      title: "ForInvest Multi-Agent Financial Intelligence API",
      version: "1.0.0",
      description:
        "Express backend endpoint layer connecting to the Python LangGraph multi-agent execution server via LangGraph JS SDK.",
    },
    servers: [{ url: `http://localhost:${PORT}` }],
  },
  // Load JSDoc annotations from the route files.
  apis: [path.join(__dirname, "routes", "*.js")],
};

module.exports = swaggerJsdoc(options);
