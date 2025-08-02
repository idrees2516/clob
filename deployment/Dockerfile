# Multi-stage build for optimized production image
FROM rust:1.75-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    libnuma-dev \
    libdpdk-dev \
    clang \
    llvm \
    && rm -rf /var/lib/apt/lists/*

# Set up build environment
WORKDIR /usr/src/app
COPY Cargo.toml Cargo.lock ./
COPY src ./src
COPY examples ./examples
COPY tests ./tests

# Configure Rust for maximum performance
ENV RUSTFLAGS="-C target-cpu=native -C target-feature=+avx2,+avx512f -C link-arg=-fuse-ld=lld"
ENV CARGO_PROFILE_RELEASE_LTO=fat
ENV CARGO_PROFILE_RELEASE_CODEGEN_UNITS=1
ENV CARGO_PROFILE_RELEASE_PANIC=abort

# Build the application
RUN cargo build --release --bin advanced_trading_system

# Production image with minimal footprint
FROM ubuntu:22.04

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libnuma1 \
    libdpdk21 \
    libssl3 \
    ca-certificates \
    numactl \
    cpufrequtils \
    ethtool \
    && rm -rf /var/lib/apt/lists/*

# Create trading user with real-time privileges
RUN groupadd -r trading && useradd -r -g trading -s /bin/bash trading

# Configure system limits for high-performance trading
RUN echo 'trading soft rtprio 99' >> /etc/security/limits.conf && \
    echo 'trading hard rtprio 99' >> /etc/security/limits.conf && \
    echo 'trading soft memlock unlimited' >> /etc/security/limits.conf && \
    echo 'trading hard memlock unlimited' >> /etc/security/limits.conf && \
    echo 'trading soft nofile 1048576' >> /etc/security/limits.conf && \
    echo 'trading hard nofile 1048576' >> /etc/security/limits.conf

# Configure huge pages
RUN echo 'vm.nr_hugepages = 2048' >> /etc/sysctl.conf && \
    echo 'vm.hugetlb_shm_group = 1000' >> /etc/sysctl.conf && \
    echo 'kernel.shmmax = 68719476736' >> /etc/sysctl.conf

# Create application directories
RUN mkdir -p /opt/trading/{bin,config,logs,data,backup} && \
    chown -R trading:trading /opt/trading

# Copy application binary
COPY --from=builder /usr/src/app/target/release/advanced_trading_system /opt/trading/bin/

# Copy configuration files
COPY deployment/config/ /opt/trading/config/
COPY deployment/scripts/ /opt/trading/scripts/

# Set up logging directory
RUN mkdir -p /var/log/trading && \
    chown trading:trading /var/log/trading

# Create mount points for persistent data
VOLUME ["/opt/trading/data", "/opt/trading/logs", "/opt/trading/backup"]

# Expose monitoring ports
EXPOSE 8080 8081 8082

# Switch to trading user
USER trading
WORKDIR /opt/trading

# Health check
HEALTHCHECK --interval=5s --timeout=3s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Set environment variables
ENV RUST_LOG=info
ENV TRADING_CONFIG_PATH=/opt/trading/config
ENV TRADING_DATA_PATH=/opt/trading/data
ENV TRADING_LOG_PATH=/opt/trading/logs

# Entry point
ENTRYPOINT ["/opt/trading/scripts/entrypoint.sh"]
CMD ["--config", "/opt/trading/config/production.toml"]