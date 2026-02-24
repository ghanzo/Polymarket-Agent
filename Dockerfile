# Stage 1: Build Polymarket CLI from source
FROM rust:1.88-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 https://github.com/Polymarket/polymarket-cli.git /build
WORKDIR /build
RUN cargo install --path . --root /usr/local

# Stage 2: Python runtime with CLI binary
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libssl3 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy the compiled Polymarket CLI binary
COPY --from=builder /usr/local/bin/polymarket /usr/local/bin/polymarket

# Set up Python app
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

CMD ["python", "-m", "src.main"]
