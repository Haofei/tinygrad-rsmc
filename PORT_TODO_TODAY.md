# Port Status — 2026-06-14

## Refactor session: adopt new rss/mc features

Rebuilt the toolchains first (the shipped binaries predated the new feature
commits): `rss` release (Jun 14) and `mcc` (Jun 14). Baseline before/after every
batch: `rss check tinygrad-rss` = **0 errors** (484→483 warnings).

### Done & verified (build stays green)

1. **`?` early-return on `Option`** — replaced manual `match {Some(x)=>…, None=>return None}`
   with `let x = expr?` at 9 genuine sites across `renderer/cstyle.rss`,
   `codegen/gpudims.rss` (4), `uop/symbolic.rss` (3), `schedule/allreduce.rss`,
   `uop/upat.rss` (3). Skipped false positives (e.g. `ArgNone` is a custom enum,
   not `Option`).
2. **mc comptime-generic `select`** — `oracle/mc_renderer.py` now emits one
   `select(comptime T: type, …)` instead of per-dtype `select_f32/f64/…` helpers;
   `mcc` monomorphizes to `select__f32`. Updated `oracle/mc_generated/relu.mc`;
   `mcc check` clean. Removed the now-dead `dedup` import.
3. **Default parameter values — keepdim collapse** — merged each reduction
   `_keepdim` overload into its base via `keepdim: Bool = false` and gated the
   final reshape-drop. Removed 6 functions
   (`tensor_{sum,max,mean}_{axis,axes}_keepdim`), updated 22 callsites + 2 mixin
   wrappers. portman coverage unchanged (these were internal helpers) — confirms
   it was a safe consolidation.

portman re-run after the collapse: symbol **51.1%**, public API **55.0%**,
weighted **43.4%**, tensor area **92.2%** — no regression.

### Investigated, NOT done — blocked by current rss feature limits

The "full sweep incl. modules / methods" hit two real limitations in this rss build:

- **Modules can't fully de-prefix.** `module` declarations give namespace
  isolation, but there is **no `use … as` aliasing and no qualified `module.fn()`
  call** — a file can bring only ONE symbol of a given name into scope. A
  collision pre-scan found **17 / 123 files** reference same-leaf functions from
  multiple owners (`main.rss` alone: 121 collisions, e.g. `cast` exists as `mk_`,
  `render_`, `tensor_`, `uop_`, `upat_`). Those files can't be de-prefixed.
  Migration cost is also ~12k callsite-token rewrites + per-file `use` injection.

- **Methods work fully (earlier "arg-position blocked" claim was WRONG).** rss has
  classes-equivalent: `struct` + `fn Type.method(self: …)` + `protocol`/`impl` +
  associated consts. An earlier note here claimed receiver-calls fail in argument
  position (RS0015) — that was a **test-harness bug** (single-line `;` statement
  separators are invalid rss). With correct multi-line syntax, method calls work in
  every position (argument, nested, return, `let` RHS, after `mut`). Conventions:
  receiver must be named `self`; `read` calls are bare `expr.method(args)`; `mut`
  calls need `mut expr.method(args)`; avoid builtin-colliding type names (`Counter`,
  `DT`, …).

  The flattening (`tensor_sum`, `uop_cast`, `mk_const`, `node_dtype`) is **not** a
  missing-class problem — it's the port's **Int-ID arena model**: a UOp/Tensor is a
  bare `Int` index into a `mut UOpCache`, so ops must be free functions threading
  the arena (UOp methods are matched by portman's `receiver_methods.UOpCache`
  config, not rss method syntax). Helpers over *real* structs (`UPat`, `DType`,
  `TensorState`, `View`, …) CAN become methods, and portman maps `fn UPat.after`
  directly onto upstream `uop/ops.py::UPat.after`.

### Done — UPat method surface (closes portman gaps)

Added thin `fn UPat.{after,end,index,reduce,sink}(self: read UPat, …)` wrappers over
the existing flat builders (`upat_after2`, …). **4 of 5 alias-needed gaps closed**
(`after`, `end`, `index`, `sink`); uop area 73.4% → **73.9%**, public API 55.0% →
**55.1%**. `UPat.reduce` stays `ambiguous` because the flat `upat_reduce` (no `2`
suffix) is already inferred as `UPat.reduce` via the `upat` owner-prefix alias, so
the explicit method makes two candidates — a portman-config nuance, not a code bug.

### Remaining method candidates (same low-risk wrapper pattern)

`DType` is already largely covered (flat `dtype_*` map via owner-prefix); the
`missing` UPat/UOp gaps (`UPat.const/alu/dtype/ufix`, `UOp.alu/const/replace/…`)
need real implementations, not wrappers. UOp aliases (`index/ranges/reduce/shard`)
are arena-Int → resolve via portman `receiver_methods` config, not rss methods.

---

# Portman snapshot

Generated with portman against freshly pulled upstream tinygrad.

- Upstream baseline: `5d5ead78dad797b0b914377d5d568efd377235ca` (2026-06-13, "inline unique_const in invalids")
- Previous baseline: `fa400f9790ab9a684387b02e958658217b33e7c1` (v0.13.0-160)
- Portman config pin updated to the new baseline in `../portman/portman.toml`.

## Current Portman Snapshot (after UPat method wrappers)

- Symbol coverage: **51.2%**
- Public API coverage: **55.1%** of 2287 API symbols
- Weighted (plan) coverage: **43.4%**
- uop area: **73.9%** (258/349)
- File coverage: **96.6%**
- Verified: **0.0%** (verification axis not wired)

Inventory: upstream **3298** symbols, target **4028** symbols.
Map: **120** file pairs (68 header-confirmed), **1674** auto-linked, **178** ambiguous name-collisions, **9** forced config links.

By status:
- implemented: 1683
- not_started: 1613
- verified: 1
- aliased: 1

By area:
- tensor    92.2%  (107/116)
- codegen   89.4%  (126/141)
- helpers   86.4%  (146/169)
- dtype     73.7%  (56/76)
- uop       73.1%  (255/349)
- engine    67.2%  (45/67)
- nn        59.9%  (85/142)
- runtime   37.5%  (460/1227)
- renderer  35.2%  (147/418)

## Upstream Upgrade Delta (`fa400f97` → `5d5ead78`)

Buckets: added **9**, removed **11**, moved **6**, signature-changed **10**, body-changed **109**.

New upstream surface to port (9):
- `tensor.py::Tensor.invalids`
- `uop/ops.py::UOp.invalids`
- `uop/ops.py::UOp.vconst_like`
- `uop/ops.py::UPat.ufix`
- `uop/ops.py::UPat._broadcasted`
- `helpers.py::to_tuple`
- `engine/realize.py::exec_hcq`
- `mixin/elementwise.py::ElementwiseMixin._uop`
- `mixin/elementwise.py::ElementwiseMixin._wrap_uop`

Candidate deprecations (upstream removed, we still implement — 3):
- `tensor.py::Tensor.ufix`
- `tensor.py::Tensor.unique_const`
- `uop/ops.py::UPat.contiguous`

Moved (6): `gradient.py::*` → `mixin/gradient.py::*` (deepwalk, reduce_gradient, compute_gradient, call_gradient, _compact_params, module body).

Re-verification worklist: **77** already-ported symbols touched by upstream changes
(includes `UOp.{bitcast,cast,call,ufix,from_buffer,_sym_fxn}`, `UPat.match`,
`tensor.py::Tensor`, `dtype.py`, `helpers.py`, several renderers). Re-verify once a
verification backend is wired.

## Public API Gaps (40: missing=22, alias_needed=12, type_only=6)

### Highest priority — UOp/UPat public surface
Top-ranked, and `UOp.alu`/`UOp.const` are also `[deps].boost` unlock symbols.

Real wrapper work (`missing`):
- `UOp.alu` (over `mk_alu1`/`mk_alu2`/`mk_alu3`)            [11]
- `UOp.const` (over `mk_const_int`/`mk_const_float`)         [11]
- `UOp.replace`, `UOp.shape`, `UOp.vconst_like`              [6]
- `UPat.alu`, `UPat.const`, `UPat.const_like`, `UPat.dtype`, `UPat.ufix`  [6]
- NEW this pull: `UOp.invalids`, `helpers.py::to_tuple`

Alias / portman-link work (code likely exists, just needs a link):
- `UOp.index`, `UOp.ranges`, `UOp.reduce`, `UOp.shard`
- `UPat.after`, `UPat.end`, `UPat.index`, `UPat.reduce`
- `TrackedPatternMatcher.rewrite`
- `bitcast`, `ssimplify`, `sym_infer`

### Follow-up batches
1. DType: `dtype.py::truncate` (type_only — decide RSS alias vs real helper).
2. Tensor training: `Tensor.train` context/class.
3. Codegen classes: `CFGContext`, `LinearScanRegallocContext`, `KernelOptError`,
   `Scheduler`, `TimeoutException`. CUDA arch constants (`cuda_sm75/80/89`) are
   type_only — track separately.
4. Engine JIT: `DepsTracker`, `GraphException`, `JitError`, `MultiGraphRunner`.
   Work only in the mirrored `tinygrad-rss/src/engine` tree.
5. Runtime/generated: keep `runtime/autogen` vendored, never hand-ported; portman
   already excludes it.

## Verification Commands

```sh
cd ../portman
PYTHONPATH=src python3 -m portman inventory
PYTHONPATH=src python3 -m portman map
PYTHONPATH=src python3 -m portman status
PYTHONPATH=src python3 -m portman gaps --public
```

Note: `portman.toml` `[verify].command` still hardcodes `/home/zoe/...` paths and
will not run on this machine — fix before relying on the verification step.

## Next Action

Start the UOp/UPat wrapper batch — `UOp.alu` and `UOp.const` first (highest rank +
dependency unlocks) — then convert the 12 `alias_needed` gaps into explicit portman
links. After that, port the 9 newly-added upstream symbols from this pull.
