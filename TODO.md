# tinygrad → rss(frontend) + mc(C backend) port — comprehensive TODO

Locked tinygrad: **v0.13.0-160-gfa400f979**. Total `tinygrad/` = 212k LOC, but **179k is
`runtime/autogen/*`** (machine-generated NVIDIA/AMD/Mesa register/ABI tables) and most of the
rest is GPU runtime drivers + non-C renderers. The **in-scope target for an rss-frontend /
mc(C)-backend port is ~10.5k LOC** (listed below).

Status key: ✅ complete · 🟡 partial (core ported, methods/behaviour remain) · ⬜ not started · ⛔ out of scope
Rule: when rss/mc lacks a feature needed for a faithful port, **fix rss/mc first**, then continue.
"Done" = every in-scope file is ✅ (and the rss/mc fixes it required are landed).

> **STATUS: IN PROGRESS — ~60% of the in-scope target.** 45 rss port files, every
> in-scope file ported at least in core and validated against tinygrad; the full lazy
> pipeline + autodiff + codegen + schedule + realize execute end-to-end via mc. rss/mc
> fixes: 11 landed (2 pending). Remaining = depth (full method/rule surfaces of the
> big files), not breadth. The port is **not 100% complete**.

---
## rss/mc fixes (land as they block the port)
- [x] rss Hashable/Eq for user types (Map/Set keys)
- [x] rss parser: parenthesized arithmetic — `f49ac9a`, suite green
- [x] rss codegen: read-param double-borrow `&&T` — `1df5c48`+`f837bf8`, suite green
- [x] mc: hosted I/O, negative float literals, float-array arithmetic, exp/log/tanh intrinsics
- [x] **mc: `UnsupportedCEmission` on nested call in a call arg** — FIXED (modern-c 05ff744): float_literal + raw.load<T> now recognized as float; zig test 142/142 green
- [x] rss: `Int.to_float` conversion — DONE (rss d9387f8): reg_vm + .rssi sig + runtime ABI; full cargo test green
- [x] rss: `let mut x = <read-param>` clone-on-bind — DONE (rss f5fab92)
- [x] rss: managed-field clone — DONE (rss 7358f50): Clone protocol + Clone.clone<T>/String.clone; full suite green; unblocked generic vec() + PtrDType
- [x] rss: `==` on a `read` enum param — DONE (rss 5dd4063)
- [x] rss: `List.get` element-type inference on struct-field lists — DONE (rss f5fab92)
- [x] rss: payload-less sum value params — DONE (rss 5dd4063)

---
## Core (top-level), ~2930 LOC
- 🟡 `dtype.py` (378) → `dtype.rss` — scalars, itemsize, predicates, can_lossless_cast, promo lattice, least_upper_dtype, **generic vec() + PtrDType** (via rss Clone). **Remaining:** ImageDType, `_get_recursive_parents` edge cases.
- 🟡 `helpers.py` (600) → `helpers.rss` — prod, ceildiv, round_up, all_same, dedup, argsort. **Remaining:** flatten, fully_flatten, get_contraction, merge_dicts, round-trip helpers, partition, getenv/Context, Timing/Profiling (host-only).
- 🟡 `device.py` (377) → `device.rss`/`device_buffer.rss` — Device[name] dispatch + CPU-interpreter backend; **Buffer lifecycle (allocate/ensure/copyin/copyout) + Allocator in_use/peak accounting** validated. **Remaining:** Compiler/CompiledProgram interface, real device backends.
- 🟡 `gradient.py` (135) → `gradient*.rss`/`autodiff*.rss`/`gradient_full.rss` — **full reverse-mode autodiff: compute_gradient reverse-toposort + accumulation, vjp for add/sub/mul/div/neg/recip/sqrt/sin/exp/log/max/where/pow** (29/29 match tinygrad .gradient()). transcendental helpers (fsqrt/fexp/fln/fsin/fcos) hand-impl to 1e-4. **Remaining (minor):** integrate with the lazy Tensor `.backward()` glue.
- 🟡 `tensor.py` (1440) → `tensor.rss`/`tensor_movement.rss`/`tensor_ops.rss` — lazy wrapper; movement+reduce shape tracking; **numeric elementwise op surface with broadcasting** (add/sub/mul/div/maximum, neg/relu/reciprocal, sum/max axis) validated vs tinygrad. + `tensor_higher.rss`: **matmul/dot, transpose, getitem (index+slice), cat/stack, softmax** (fexp via range-reduced Taylor) validated vs tinygrad. + **`tensor_full.rss`**: creation ops (full/zeros/ones/arange/eye), reshape/flatten, mean(axis), and an end-to-end **.backward()** linreg demo (dL/dw=-110, dL/db=-38 match tinygrad; SGD step lowers loss). + **`tensor_conv.rss`**: conv2d (single + multichannel), max_pool2d, avg_pool2d (match tinygrad). **Remaining:** randn (RNG), fancy/boolean indexing, strided/padded conv.

## uop/, ~3711 LOC
- 🟡 `uop/ops.py` (1708) → `uop.rss`,`rewrite.rss`,`upat.rss`,`symbolic.rss`,`symbolic_rules.rss`,`toposort.rss`,`ops_enum.rss`,`uop_methods.rss` — UOp node + interning, PatternMatcher + UPat DSL, vmin/vmax, toposort, full 93-op enum+groups, **UOp method surface (substitute/const_like/ufix/broadcast/replace/key/vmin-vmax)**. + **`graph_rewrite.rss`**: generic rules-as-data fixpoint engine (PatternMatcher.rewrite + RewriteContext + memoized bottom-up graph_rewrite) — rules are data, driver is generic (matches rewrite.rss + tinygrad). **Remaining:** sint helpers, exhaustive symbolic ruleset.
- 🟡 `uop/symbolic.py` (485) → `symbolic.rss` (vmin/vmax only) — **Remaining:** the symbolic algebra simplification rules (the bulk).
- 🟡 `uop/upat.py` (168) → `upat.rss` — core recursive matcher + capture. **Remaining:** named captures, sets of ops, location/dtype constraints, allow_any_len.
- 🟡 `uop/render.py` (159) → `uop_render.rss` — **UOp→infix string render** (Const/Var/Add/Sub/Mul/Div/Mod, parenthesized): ((v0+2)*3), (((v0*4)+v1)%8). **Remaining:** precedence-aware paren elision, dtype/special-op rendering.
- 🟡 `uop/divandmod.py` (116) → `divandmod.rss` — **floor div/mod (Python semantics) + folding rules** (const-fold, x//1, x%1, (x*c)//c, (x*c)%c, c|lm: (x*lm+r)//c→x*(lm/c)+r//c, %→r%c); matches tinygrad. **Remaining:** gcd_with_remainder, congruence folding, nested mod.
- 🟡 `uop/decompositions.py` (572) → `decompositions.rss` — **transcendental decompositions as UOp rewrites** (EXP→EXP2·log2e, LOG→LOG2·ln2, POW→exp2(log2·b), TAN→sin/sin(x+π/2)), constants match tinygrad. **Remaining:** full ID set, range reduction for sin, fp edge cases.
- 🟡 `uop/spec.py` (280) + `validate.py` (78) → `spec.rss` — **structural validation**: per-op arity table + ALU same-dtype + WHERE bool-cond checks, recursive over the graph (valid/invalid cases match). **Remaining:** full per-op dtype/shape spec rules, z3 symbolic bound checking.
- 🟡 `uop/__init__.py` (145) → covered by `ops_enum.rss` — its substance is `FastEnum` + the `Ops` enum (all 93 ops ported). **Remaining:** FastEnum auto-numbering is a Python metaclass detail, N/A for rss.

## codegen/, ~1000 LOC in-scope (opt-search excluded)
- 🟡 lowering to kernels → `render*.rss` (elementwise/reduce/matmul + relu/div/neg/max/exp/log/tanh emitters), `buffer_reuse.rss` (liveness memory planner), **`linearizer.rss`** (toposort + priority-heap ordering -> SSA instruction stream, RANGE/ENDRANGE placement; matches tinygrad ordering). + **`codegen_late.rss`**: expander (no_vectorized_alu: VEC ALU -> N scalar lanes) + devectorizer folds (GEP-of-VEC -> lane, VEC-of-same -> broadcast), cross-checked vs tinygrad STACK/gep. **Remaining:** `simplify.py` full ruleset, `late/regalloc.py`, WMMA/UNROLL axis bookkeeping, gpudims (GPU-only).
- ⛔ `codegen/opt/*` (search/heuristic/tc/postrange ~875) — autotuning search; out of scope for a correctness port.

## renderer/, in-scope = cstyle only
- 🟡 `renderer/cstyle.py` (572) → `cstyle_renderer.rss` — **per-UOp render map over a lowered kernel arena (LOAD/CONST/ALU/STORE in a RANGE), compiles+runs via mc** ((x+2)*3 over a buffer). **Remaining:** DEFINE_GLOBAL/SPECIAL/index render, dtype-aware casts, full op render map, multi-output.
- ⛔ `renderer/{amd,ptx,llvmir,nir,wgsl,isa}` (~3.9k) — other backends, not the mc/C target.

## engine/, ~265 in-scope
- 🟡 `engine/realize.py` (265) → `realize.rss`,`schedule*.rss`,`device.rss` — **ExecItem list + realize(): schedule a graph to ordered kernels, allocate buffers, run in order via mc** (x-sum(x) -> 2 kernels K1(reduce)->intermediate->K2(map), executes natively, matches tinygrad). **Remaining:** CompiledRunner/BufferCopy abstraction, var binding, JIT.
- ⛔ `engine/jit.py` (312) — TinyJit graph capture/replay (optimization; out of scope for correctness).

## schedule/, ~1100 in-scope
- 🟡 `schedule/__init__.py` (147) + `memory.py` (64) + `rangeify.py` (611) → `schedule.rss`,`schedule_multi.rss`,`buffer_reuse.rss`,`rangeify.rss` — kernel-shape dispatch, reduce-split, liveness buffer reuse, **kernel-grouping/fusion pass** (elementwise fuse, reduce/store/shared boundaries; kernel counts match tinygrad). + **`indexing.rss`**: index/valid generation -- map output multi-index through movement chain (permute/reshape) to input offset + PAD valid mask (matches tinygrad). **Remaining:** COPY/upload kernels, cost model.
- ⛔ `schedule/multi.py` (175), `allreduce.py` (62) — multi-GPU, out of scope.

## nn/, ~900 in-scope
- 🟡 `nn/optim.py` (179) → `train.rss`,`nn_layers.rss` — SGD step (to convergence) + **SGD/Adam update steps validated vs tinygrad** (`nn_layers.rss`). **Remaining:** momentum/weight-decay, AdamW/LARS as stateful classes.
- 🟡 `nn/__init__.py` (419) → `nn_layers.rss` — **Linear forward, relu, layernorm validated vs tinygrad** (numeric, fsqrt via Newton). **Remaining:** Conv2d, BatchNorm, Embedding as classes.
- 🟡 `nn/state.py` (294) → `nn_state.rss` — **get_state_dict/load_state_dict** core: named params -> flat (names,data,lens), get_param by index, load round-trip (validated). **Remaining:** safetensors/torch_load binary formats (host I/O).
- ⛔ `nn/onnx.py` (1314) — ONNX import; out of scope.

---
## Capstones (cross-cutting, validated, executing via mc)
- [x] full lazy pipeline Tensor→UOp→simplify→render→native ((x+2)*3, sum, matmul)
- [x] SGD training to convergence (`train.rss`)

## Honest completion accounting (latest)
**NOT finished.** 32 rss files in `port-rss/` run and validate against tinygrad, covering a
working core of every in-scope subsystem — but **no large file is ✅ fully complete** (tensor.py
higher ops, uop/ops.py full rule set, codegen pipeline, schedule/rangeify all remain partial).
Rough completion of the ~10.5k in-scope target: **~40%**.

Biggest remaining chunks (in rough priority):
1. `schedule/rangeify.py` (611) — the real fusion/range engine ⬜
2. `tensor.py` higher ops — matmul/conv/indexing/creation/`.backward()`+`.realize()` glue 🟡
3. `codegen/` pipeline — linearizer/devectorizer/expander/simplify 🟡
4. `uop/ops.py` full method surface + the rest of `symbolic.py` 🟡
5. `uop/{spec,validate,divandmod,render}` ⬜, `nn/state.py` ⬜, `schedule/indexing.py` ⬜
6. engine realize end-to-end (#12) ⬜
7. pending rss fixes: Int→Float, managed-field clone, let-mut clone, read-enum `==`, field-list get inference, sum-type value params

"Port done" = all 🟡/⬜ above turned ✅ with the listed rss/mc fixes landed.
