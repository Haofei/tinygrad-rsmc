# Vendored tinygrad runtime autogen

This directory contains generated runtime binding data copied from upstream tinygrad.

- Source tree: `/home/zoe/tinygrad/tinygrad/runtime/autogen`
- Upstream commit at copy time: `fa400f9790ab9a684387b02e958658217b33e7c1`
- Vendored path: `tinygrad-rss/vendor/tinygrad/runtime/autogen`
- File set: 88 Python source files, 179323 LOC
- Sorted file-list SHA256: `4b712b26144f947976747a91df53ec51c6efe68f5da0fdaf91e07031d1625586`
- Sorted per-file SHA256 manifest SHA256: `1b4e66b393d4516a3a28e9eefcb5c94e64eec236b86c89a4f4cfc2f4da12d38b`

These files are generated data. They are intentionally vendored rather than hand-ported.
Do not add this directory to `rsspkg.toml` sources; handwritten runtime code should import or
consume the generated data once the runtime/device layer is ported.
