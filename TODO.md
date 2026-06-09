# tinygrad → rss(frontend) + mc(C backend) port — comprehensive TODO

Locked tinygrad: **v0.13.0-160-gfa400f979**. Total `tinygrad/` = 212k LOC, but **179k is
`runtime/autogen/*`** (machine-generated NVIDIA/AMD/Mesa register/ABI tables) and most of the
rest is GPU runtime drivers + non-C renderers. The **in-scope target for an rss-frontend /
mc(C)-backend port is ~10.5k LOC** (listed below).

Status key: ✅ complete · 🟡 partial (core ported, methods/behaviour remain) · ⬜ not started · ⛔ out of scope
Rule: when rss/mc lacks a feature needed for a faithful port, **fix rss/mc first**, then continue.
"Done" = every in-scope file is ✅ (and the rss/mc fixes it required are landed).

---
## rss/mc fixes (land as they block the port)
- [x] rss Hashable/Eq for user types (Map/Set keys)
- [x] rss parser: parenthesized arithmetic — `f49ac9a`, suite green
- [x] rss codegen: read-param double-borrow `&&T` — `1df5c48`+`f837bf8`, suite green
- [x] mc: hosted I/O, negative float literals, float-array arithmetic, exp/log/tanh intrinsics
- [ ] **mc: `UnsupportedCEmission` on nested call in a sequenced call arg** (e.g. `raw.store(.., (raw.load(..)+2)*3)`; inner `i*4` lacks target type) — blocks inline cstyle; lower_c.zig:3451
- [ ] rss: `Int.to_float` / `Float.from_int` conversion (deep: reg_vm + builtin-interface sig + lowering ABI + runtime)
- [ ] rss: `let mut x = <read-param>` clone-on-bind (codegen E0308)
- [ ] rss: managed-field clone (callable `.clone()`/take-rebuild) — blocks generic `vec()`, `PtrDType`
- [ ] rss: `==` on a `read` enum param (auto-deref)
- [ ] rss: `List.get` element-type inference on struct-field lists
- [ ] rss: sum-type value params (pass an enum variant by value to a param)

---
## Core (top-level), ~2930 LOC
- 🟡 `dtype.py` (378) → `dtype.rss` — scalars, itemsize(count-aware), predicates, can_lossless_cast, promo lattice, least_upper_dtype, vec(). **Remaining:** PtrDType, ImageDType (blocked: managed-field clone), `_get_recursive_parents` edge cases.
- 🟡 `helpers.py` (600) → `helpers.rss` — prod, ceildiv, round_up, all_same, dedup, argsort. **Remaining:** flatten, fully_flatten, get_contraction, merge_dicts, round-trip helpers, partition, getenv/Context, Timing/Profiling (host-only).
- 🟡 `device.py` (377) → `device.rss` — Device[name] dispatch + CPU-interpreter backend, host Float buffers. **Remaining:** Allocator/Buffer lifecycle, Compiler/CompiledProgram interface, Compiled/MallocAllocator.
- 🟡 `gradient.py` (135) → `gradient.rss`/`autodiff_dag.rss`/`gradient_relu.rss`/`autodiff_ops.rss` — reverse-mode (tree + DAG accumulation), vjp for add/mul/sub/relu/div/neg/exp/log. **Remaining:** vjp for the rest (recip/sqrt/sin/where/cmplt/max/pow), the actual `compute_gradient` graph-walk over real UOps + accumulate via toposort.
- 🟡 `tensor.py` (1440) → `tensor.rss`/`tensor_movement.rss` — lazy UOp-graph wrapper; movement (reshape/permute/expand) + reduce (sum) with shape tracking; elementwise add/mul/sub. **Remaining (large):** the hundreds of Tensor methods — creation (full/randn/arange/eye), binary/unary op surface, matmul/dot/conv2d, indexing/getitem, pad/cat/stack, softmax/etc., `.backward()`, `.realize()`, broadcasting rules.

## uop/, ~3711 LOC
- 🟡 `uop/ops.py` (1708) → `uop.rss`,`rewrite.rss`,`upat.rss`,`symbolic.rss`,`toposort.rss`,`ops_enum.rss` — UOp node + interning, PatternMatcher rewrite, generic UPat DSL, vmin/vmax, toposort, **complete 93-op Ops enum + GroupOp groups**. **Remaining:** the full UOp method/property surface, the big `symbolic`/`sym` PatternMatcher rule set, graph_rewrite engine details, sint helpers.
- 🟡 `uop/symbolic.py` (485) → `symbolic.rss` (vmin/vmax only) — **Remaining:** the symbolic algebra simplification rules (the bulk).
- 🟡 `uop/upat.py` (168) → `upat.rss` — core recursive matcher + capture. **Remaining:** named captures, sets of ops, location/dtype constraints, allow_any_len.
- ⬜ `uop/render.py` (159) — UOp graph → string render helpers (overlaps cstyle).
- ⬜ `uop/divandmod.py` (116) — div/mod symbolic folding rules.
- ⬜ `uop/decompositions.py` (572) — op decompositions (e.g. transcendental → exp2/log2 sequences).
- ⬜ `uop/spec.py` (280) — UOp validation spec (type/shape constraints per op).
- ⬜ `uop/validate.py` (78) — graph validation.
- ⬜ `uop/__init__.py` (145).

## codegen/, ~1000 LOC in-scope (opt-search excluded)
- 🟡 lowering to kernels → `render*.rss` (elementwise/reduce/matmul + relu/div/neg/max/exp/log/tanh emitters), `buffer_reuse.rss` (liveness memory planner). **Remaining:** faithful `codegen/__init__.py` (236) pipeline, `simplify.py` (157), `late/linearizer.py` (96), `late/devectorizer.py` (390), `late/expander.py` (160), `late/regalloc.py` (137), `gpudims.py` (111, GPU-only — partial scope).
- ⛔ `codegen/opt/*` (search/heuristic/tc/postrange ~875) — autotuning search; out of scope for a correctness port.

## renderer/, in-scope = cstyle only
- 🟡 `renderer/cstyle.py` (572) → `cstyle_renderer.rss` (IN PROGRESS) — per-UOp render map over a lowered kernel arena (LOAD/CONST/ALU/STORE in a RANGE). **Remaining:** name-per-UOp temps (faithful + dodges the mc nested-call bug), DEFINE_GLOBAL/SPECIAL/index render, dtype-aware casts, full op render map, the C prologue/struct.
- ⛔ `renderer/{amd,ptx,llvmir,nir,wgsl,isa}` (~3.9k) — other backends, not the mc/C target.

## engine/, ~265 in-scope
- 🟡 `engine/realize.py` (265) → `schedule*.rss`,`device.rss` — single+multi-kernel scheduling/dispatch, run via mc. **Remaining:** faithful ExecItem/CompiledRunner/BufferCopy, Schedule execution order, var binding.
- ⛔ `engine/jit.py` (312) — TinyJit graph capture/replay (optimization; out of scope for correctness).

## schedule/, ~1100 in-scope
- 🟡 `schedule/__init__.py` (147) + `memory.py` (64) → `schedule.rss`,`schedule_multi.rss`,`buffer_reuse.rss` — kernel-shape dispatch, reduce-split, liveness buffer reuse. **Remaining:** faithful `rangeify.py` (611, the real fusion/range engine), `indexing.py` (286, index/valid generation).
- ⛔ `schedule/multi.py` (175), `allreduce.py` (62) — multi-GPU, out of scope.

## nn/, ~900 in-scope
- 🟡 `nn/optim.py` (179) → `train.rss`,`nn_layers.rss` — SGD step (to convergence) + **SGD/Adam update steps validated vs tinygrad** (`nn_layers.rss`). **Remaining:** momentum/weight-decay, AdamW/LARS as stateful classes.
- 🟡 `nn/__init__.py` (419) → `nn_layers.rss` — **Linear forward, relu, layernorm validated vs tinygrad** (numeric, fsqrt via Newton). **Remaining:** Conv2d, BatchNorm, Embedding as classes.
- ⬜ `nn/state.py` (294) — get_state_dict/load_state_dict/safetensors (host serialization).
- ⛔ `nn/onnx.py` (1314) — ONNX import; out of scope.

---
## Capstones (cross-cutting, validated, executing via mc)
- [x] full lazy pipeline Tensor→UOp→simplify→render→native ((x+2)*3, sum, matmul)
- [x] SGD training to convergence (`train.rss`)

## Honest completion accounting
Every in-scope subsystem has a **validated core (🟡)**; **none of the large files (tensor.py,
uop/ops.py, codegen, schedule/rangeify) is ✅ complete** — the full method/rule sets remain.
Rough completion of the ~10.5k in-scope target: **~25–30%**. "Port done" = all 🟡/⬜ above
turned ✅, with the listed rss/mc fixes landed.
