#!/usr/bin/env bash
# Build the rss + mcc toolchains inside the dev container.
#
# Idempotent: cargo/zig do incremental rebuilds, and the outputs land in the
# container-local volumes declared in docker-compose.yml, so this is cheap to
# re-run and survives across `docker compose run` invocations. Run it once after
# `docker compose build`, and again whenever the sibling toolchain sources change.
set -euo pipefail

RSSCRIPT_DIR="${RSSCRIPT_DIR:-/work/rsscript}"
MC_DIR="${MC_DIR:-/work/modern-c}"

echo ">> Building rss (RSScript compiler) — cargo build --release"
( cd "$RSSCRIPT_DIR" && cargo build --release )

echo ">> Building mcc (modern-c compiler) — zig build"
( cd "$MC_DIR" && zig build )

echo
echo ">> Toolchain versions:"
"$RSSCRIPT_DIR/target/release/rss" --help >/dev/null 2>&1 && echo "   rss:  ok ($RSSCRIPT_DIR/target/release/rss)"
"$MC_DIR/zig-out/bin/mcc" 2>/dev/null | head -1 >/dev/null 2>&1 || true
echo "   mcc:  $("$MC_DIR/zig-out/bin/mcc" --help 2>&1 | head -1 || echo present)"
echo "   zig:  $(zig version)"
echo "   $(python3 --version)"
echo "   $(clang --version | head -1)"
echo
echo ">> Toolchains ready. Try: rss check tinygrad-rss"
