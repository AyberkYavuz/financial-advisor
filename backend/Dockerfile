FROM node:20-alpine

WORKDIR /app

# Copy dependency manifests first for layer caching.
COPY package.json package-lock.json ./
RUN npm ci

# Copy source.
COPY server.js ./
COPY routes ./routes
COPY swagger.js ./

EXPOSE 3000

CMD ["npm", "start"]
