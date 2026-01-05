# Dockerfile for Supervisor sandbox containers
# Two images: cli (with network egress) and executor (no network)

# Base image with common tools
FROM python:3.11-slim AS base

# Install common dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -s /bin/bash worker
WORKDIR /workspace

# -------------------------------------------
# CLI Image: For running AI CLIs (needs network for API calls)
# -------------------------------------------
FROM base AS cli

# Install Node.js for some CLIs
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Install Claude CLI (example - adjust based on actual installation method)
# RUN npm install -g @anthropic-ai/claude-cli

# Install Python tools
RUN pip install --no-cache-dir \
    anthropic \
    openai \
    google-generativeai

# Copy CLI wrapper scripts
COPY scripts/cli_wrapper.sh /usr/local/bin/cli_wrapper
RUN chmod +x /usr/local/bin/cli_wrapper

USER worker
ENTRYPOINT ["/usr/local/bin/cli_wrapper"]

# -------------------------------------------
# Executor Image: For running tests/commands (no network)
# -------------------------------------------
FROM base AS executor

# Install common build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    make \
    && rm -rf /var/lib/apt/lists/*

# Install Python testing tools
RUN pip install --no-cache-dir \
    pytest \
    pytest-cov \
    ruff \
    mypy

USER worker
ENTRYPOINT ["/bin/bash", "-c"]
