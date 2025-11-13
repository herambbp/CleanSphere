# Multi-stage build for optimized production image

# Stage 1: Build frontend
FROM node:18-alpine AS frontend-build
WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm ci

# Copy source code
COPY . .

# Build frontend
RUN npm run build

# Stage 2: Production image
FROM node:18-alpine AS production
WORKDIR /app

# Copy package files
COPY package*.json ./

# Install only production dependencies
RUN npm ci --only=production

# Copy built frontend from previous stage
COPY --from=frontend-build /app/dist ./dist

# Copy server code
COPY server ./server

# Copy notes
COPY notes ./notes

# Expose port
EXPOSE 5000

# Set environment variables
ENV NODE_ENV=production

# Start server
CMD ["node", "server/index.js"]
