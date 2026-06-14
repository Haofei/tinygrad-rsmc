# Combined cross-platform development image for the tinygrad-rsmc port.
#
# The port is produced by THREE toolchains that live in sibling repos:
#   - rss  (RSScript compiler) from ../rsscript  — Rust  (checks/builds tinygrad-rss)
#   - mcc  (modern-c compiler) from ../modern-c  — Zig   (the codegen backend target)
#   - portman + oracle                            — Python 3.11 + clang
#
# Following the sibling repos' convention, the source is NOT copied into the
# image: the whole workspace is bind-mounted at /work (see docker-compose.yml),
# so edits on the host are live on every platform. This image is JUST the
# toolchain. Multi-arch: builds natively on linux/amd64 and linux/arm64.
#
# rust:1-bookworm gives Rust (>= 1.85, the rsscript workspace edition needs it)
# on Debian 12, whose python3 is 3.11 — exactly what portman (stdlib tomllib)
# and tinygrad (requires-python >= 3.11) need, so one base covers everything.
FROM rust:1-bookworm

# Pin to the Zig version modern-c's CI/Docker uses.
ARG ZIG_VERSION=0.16.0
ENV DEBIAN_FRONTEND=noninteractive

# System toolchain:
#   build-essential, pkg-config, cmake -> build rsscript (ring/rusqlite) + gcc
#   clang, lld, llvm                   -> mcc's C backend, the oracle's `clang -lm`,
#                                         and tinygrad's CPU backend (it shells out to clang)
#   python3, python3-numpy             -> portman (stdlib, >= 3.11) + oracle/tinygrad
#   git/curl/wget/xz-utils/ca-certs/bash/make -> fetch + build + run helpers
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        build-essential pkg-config cmake \
        clang lld llvm \
        python3 python3-numpy \
        git curl wget xz-utils ca-certificates bash make \
 && rm -rf /var/lib/apt/lists/*

# Rust components used by the rss lint/test flow.
RUN rustup component add clippy rustfmt

# Zig: fetch the exact release tarball for this build's architecture from
# ziglang.org's index.json (arch-agnostic), so the same Dockerfile works on
# amd64 and arm64 (Apple Silicon). Mirrors modern-c/Dockerfile.
RUN set -eux; \
    case "$(uname -m)" in \
        x86_64) zarch=x86_64 ;; \
        aarch64|arm64) zarch=aarch64 ;; \
        *) echo "unsupported architecture: $(uname -m)" >&2; exit 1 ;; \
    esac; \
    url="$(python3 -c "import json,urllib.request; d=json.load(urllib.request.urlopen('https://ziglang.org/download/index.json')); print(d['${ZIG_VERSION}']['${zarch}-linux']['tarball'])")"; \
    wget -qO /tmp/zig.tar.xz "$url"; \
    mkdir -p /opt/zig; \
    tar -xJf /tmp/zig.tar.xz -C /opt/zig --strip-components=1; \
    ln -sf /opt/zig/zig /usr/local/bin/zig; \
    rm /tmp/zig.tar.xz; \
    zig version

# Put the built toolchain binaries on PATH for every shell flavor (login shells
# in some devcontainer setups re-derive PATH from /etc/profile).
RUN printf 'export PATH="/work/rsscript/target/release:/work/modern-c/zig-out/bin:%s/bin:$PATH"\n' \
        "${CARGO_HOME:-/usr/local/cargo}" > /etc/profile.d/rsmc-toolchain.sh

# The built rss/mcc binaries (produced by scripts/docker/bootstrap.sh into the
# bind-mounted sibling trees) and the rss runtime crate need to be discoverable.
ENV PATH="/work/rsscript/target/release:/work/modern-c/zig-out/bin:${PATH}" \
    RSSCRIPT_RUNTIME_PATH="/work/rsscript/crates/runtime" \
    MC_ROOT="/work/modern-c" \
    PYTHONPATH="/work/tinygrad"

WORKDIR /work/tinygrad-rsmc

# Default to an interactive shell; compose/Make override for one-off commands.
CMD ["bash"]
