FROM node:20-alpine

# Create app directory
WORKDIR /app

# Copy package files first for layer caching
COPY package*.json ./

# Install only production dependencies
RUN npm ci --only=production

# Copy source code
COPY src/ ./src/

# Create uploads directory (Railway uses ephemeral storage, but needed for runtime)
RUN mkdir -p uploads

# Expose port (Railway sets PORT env automatically)
EXPOSE 8080

# Start the server
CMD ["node", "src/server.js"]
