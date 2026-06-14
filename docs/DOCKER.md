# Dockerized development

Develop the `tinygrad-rsmc` port on any platform (Linux, macOS Intel/Apple
Silicon, Windows/WSL2) with one container that bundles every toolchain the port
needs. No host Rust/Zig/Python/clang install required — only Docker.

## What's in the image

The port is produced by three toolchains that live in **sibling repos**, plus a
Python layer. The dev image (`Dockerfile`) bundles all of them:

| Tool | From | Used for |
|---|---|---|
| `rss` (RSScript compiler) | `../rsscript` (Rust) | check/build `tinygrad-rss` |
| `mcc` (modern-c compiler) | `../modern-c` (Zig 0.16.0) | the codegen backend target |
| `portman` | `../portman` (Python 3.11) | upstream port coverage tracking |
| oracle | `oracle/` (Python + clang) | numerical C-reference proof |

Following the convention of `rsscript` and `modern-c`, the image is
**toolchain-only**: source is *not* copied in. The repos are **bind-mounted
live**, so edits on the host are seen instantly inside the container.

## Prerequisites

Docker (with Compose v2) and the sibling repos checked out **next to** this one —
the same layout `portman.toml` already assumes:

```
work/
├── tinygrad-rsmc/   ← you are here
├── rsscript/
├── modern-c/
├── tinygrad/        ← upstream tinygrad (the port source of truth)
└── portman/
```

## Quickstart

From the repo root:

```bash
make build        # build the toolchain image (once; ~minutes, pulls + installs)
make bootstrap    # compile rss + mcc inside the container (incremental after)
make check        # rss check tinygrad-rss   (expect 0 errors)
make portman      # portman inventory + map + status
make oracle       # C-reference numerical roundtrip
make shell        # interactive shell in the container
```

`make help` lists every target. Equivalent raw Compose commands:

```bash
docker compose build dev
docker compose run --rm dev scripts/docker/bootstrap.sh
docker compose run --rm dev rss check tinygrad-rss
docker compose run --rm dev                      # interactive shell
```

## How it's wired

- **Live source, container-local artifacts.** The whole workspace is
  bind-mounted, but architecture-specific build outputs — `rsscript/target`,
  `modern-c/zig-out`, the Zig caches, and the cargo registry — are container-local
  **named volumes**. A host `cargo build` (macOS arm64 Mach-O) and a container
  build (linux ELF) therefore never clobber each other.
- **`scripts/docker/bootstrap.sh`** compiles `rss` and `mcc` into those volumes.
  It's idempotent and incremental; re-run it after changing the sibling toolchain
  sources. The built binaries are on `PATH` (`rss`, `mcc`) in every shell.
- **Pre-set env** (in `docker-compose.yml` and the image): `RSSCRIPT_RUNTIME_PATH`
  (rss runtime crate), `MC_ROOT` (modern-c root the oracle invokes `mcc` from),
  and `PYTHONPATH` (upstream tinygrad). The oracle and `portman.toml` use
  workspace-relative paths, so they also work outside Docker on the standard
  layout.

## Resetting

```bash
make clean        # docker compose down -v — drop the build volumes
```

This removes the cached Rust/Zig artifacts; the next `make bootstrap` rebuilds
the toolchains from scratch.

## Multi-arch

The image builds natively on `linux/amd64` and `linux/arm64` (Apple Silicon): the
Rust base is multi-arch and the Zig release is selected per-architecture at build
time, matching `modern-c`'s pinned `0.16.0`.
