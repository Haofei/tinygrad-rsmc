# tinygrad → rss(frontend) + mc(C backend) port — comprehensive TODO

Locked tinygrad: **v0.13.0-160-gfa400f979**. Total `tinygrad/` = 212k LOC, but **179k is
`runtime/autogen/*`** (machine-generated NVIDIA/AMD/Mesa register/ABI tables) and most of the
rest is GPU runtime drivers + non-C renderers. The **in-scope target for an rss-frontend /
mc(C)-backend port is ~10.5k LOC** (listed below).

Status key: ✅ complete · 🟡 partial (core ported, methods/behaviour remain) · ⬜ not started · ⛔ out of scope
Rule: when rss/mc lacks a feature needed for a faithful port, **fix rss/mc first**, then continue.
"Done" = every in-scope file is ✅ (and the rss/mc fixes it required are landed).

> **STATUS: NOT COMPLETE — ~45% of the in-scope target.** 37 rss port files exist and
> run/validate against tinygrad, but **no large file is ✅ fully complete** (cores done, full
> method/rule sets remain). Tally: 0 ✅ · 18 🟡 partial · 2 ⬜ not started · ⛔ as marked.
> rss/mc fixes: 5 landed (incl. mc nested-call) · Int→Float verifying · ~5 pending. The items
> below are the remaining work; the port is **not finished**.

---
## rss/mc fixes (land as they block the port)
- [x] rss Hashable/Eq for user types (Map/Set keys)
- [x] rss parser: parenthesized arithmetic — `f49ac9a`, suite green
- [x] rss codegen: read-param double-borrow `&&T` — `1df5c48`+`f837bf8`, suite green
- [x] mc: hosted I/O, negative float literals, float-array arithmetic, exp/log/tanh intrinsics
- [x] **mc: `UnsupportedCEmission` on nested call in a call arg** — FIXED (modern-c 05ff744): float_literal + raw.load<T> now recognized as float; zig test 142/142 green
- [x] rss: `Int.to_float` conversion — DONE (rss d9387f8): reg_vm + .rssi sig + runtime ABI; full cargo test green
- [ ] rss: `let mut x = <read-param>` clone-on-bind (codegen E0308)
- [ ] rss: managed-field clone (callable `.clone()`/take-rebuild) — blocks generic `vec()`, `PtrDType`
- [ ] rss: `==` on a `read` enum param (auto-deref)
- [ ] rss: `List.get` element-type inference on struct-field lists
- [ ] rss: sum-type value params (pass an enum variant by value to a param)

---
## Core (top-level), ~2930 LOC
- 🟡 `dtype.py` (378) → `dtype.rss` — scalars, itemsize(count-aware), predicates, can_lossless_cast, promo lattice, least_upper_dtype, vec(). **Remaining:** PtrDType, ImageDType (blocked: managed-field clone), `_get_recursive_parents` edge cases.
- 🟡 `helpers.py` (600) → `helpers.rss` — prod, ceildiv, round_up, all_same, dedup, argsort. **Remaining:** flatten, fully_flatten, get_contraction, merge_dicts, round-trip helpers, partition, getenv/Context, Timing/Profiling (host-only).
- 🟡 `device.py` (377) → `device.rss`/`device_buffer.rss` — Device[name] dispatch + CPU-interpreter backend; **Buffer lifecycle (allocate/ensure/copyin/copyout) + Allocator in_use/peak accounting** validated. **Remaining:** Compiler/CompiledProgram interface, real device backends.
- 🟡 `gradient.py` (135) → `gradient.rss`/`autodiff_dag.rss`/`gradient_relu.rss`/`autodiff_ops.rss` — reverse-mode (tree + DAG accumulation), vjp for add/mul/sub/relu/div/neg/exp/log. **Remaining:** vjp for the rest (recip/sqrt/sin/where/cmplt/max/pow), the actual `compute_gradient` graph-walk over real UOps + accumulate via toposort.
- 🟡 `tensor.py` (1440) → `tensor.rss`/`tensor_movement.rss`/`tensor_ops.rss` — lazy wrapper; movement+reduce shape tracking; **numeric elementwise op surface with broadcasting** (add/sub/mul/div/maximum, neg/relu/reciprocal, sum/max axis) validated vs tinygrad. + `tensor_higher.rss`: **matmul/dot, transpose, getitem (index+slice), cat/stack, softmax** (fexp via range-reduced Taylor) validated vs tinygrad. **Remaining:** conv2d, creation ops (randn/arange/eye), `.backward()`/`.realize()` glue.

## uop/, ~3711 LOC
- 🟡 `uop/ops.py` (1708) → `uop.rss`,`rewrite.rss`,`upat.rss`,`symbolic.rss`,`toposort.rss`,`ops_enum.rss` — UOp node + interning, PatternMatcher rewrite, generic UPat DSL, vmin/vmax, toposort, **complete 93-op Ops enum + GroupOp groups**. **Remaining:** the full UOp method/property surface, the big `symbolic`/`sym` PatternMatcher rule set, graph_rewrite engine details, sint helpers.
- 🟡 `uop/symbolic.py` (485) → `symbolic.rss` (vmin/vmax only) — **Remaining:** the symbolic algebra simplification rules (the bulk).
- 🟡 `uop/upat.py` (168) → `upat.rss` — core recursive matcher + capture. **Remaining:** named captures, sets of ops, location/dtype constraints, allow_any_len.
- 🟡 `uop/render.py` (159) → `uop_render.rss` — **UOp→infix string render** (Const/Var/Add/Sub/Mul/Div/Mod, parenthesized): ((v0+2)*3), (((v0*4)+v1)%8). **Remaining:** precedence-aware paren elision, dtype/special-op rendering.
- 🟡 `uop/divandmod.py` (116) → `divandmod.rss` — **floor div/mod (Python semantics) + folding rules** (const-fold, x//1, x%1, (x*c)//c, (x*c)%c, c|lm: (x*lm+r)//c→x*(lm/c)+r//c, %→r%c); matches tinygrad. **Remaining:** gcd_with_remainder, congruence folding, nested mod.
- 🟡 `uop/decompositions.py` (572) → `decompositions.rss` — **transcendental decompositions as UOp rewrites** (EXP→EXP2·log2e, LOG→LOG2·ln2, POW→exp2(log2·b), TAN→sin/sin(x+π/2)), constants match tinygrad. **Remaining:** full ID set, range reduction for sin, fp edge cases.
- 🟡 `uop/spec.py` (280) + `validate.py` (78) → `spec.rss` — **structural validation**: per-op arity table + ALU same-dtype + WHERE bool-cond checks, recursive over the graph (valid/invalid cases match). **Remaining:** full per-op dtype/shape spec rules, z3 symbolic bound checking.
- ⬜ `uop/__init__.py` (145).

## codegen/, ~1000 LOC in-scope (opt-search excluded)
- 🟡 lowering to kernels → `render*.rss` (elementwise/reduce/matmul + relu/div/neg/max/exp/log/tanh emitters), `buffer_reuse.rss` (liveness memory planner), **`linearizer.rss`** (toposort + priority-heap ordering -> SSA instruction stream, RANGE/ENDRANGE placement; matches tinygrad ordering). **Remaining:** `simplify.py` (157), `late/devectorizer.py` (390), `late/expander.py` (160), `late/regalloc.py` (137), gpudims (GPU-only).
- ⛔ `codegen/opt/*` (search/heuristic/tc/postrange ~875) — autotuning search; out of scope for a correctness port.

## renderer/, in-scope = cstyle only
- 🟡 `renderer/cstyle.py` (572) → `cstyle_renderer.rss` — **per-UOp render map over a lowered kernel arena (LOAD/CONST/ALU/STORE in a RANGE), compiles+runs via mc** ((x+2)*3 over a buffer). **Remaining:** DEFINE_GLOBAL/SPECIAL/index render, dtype-aware casts, full op render map, multi-output.
- ⛔ `renderer/{amd,ptx,llvmir,nir,wgsl,isa}` (~3.9k) — other backends, not the mc/C target.

## engine/, ~265 in-scope
- 🟡 `engine/realize.py` (265) → `realize.rss`,`schedule*.rss`,`device.rss` — **ExecItem list + realize(): schedule a graph to ordered kernels, allocate buffers, run in order via mc** (x-sum(x) -> 2 kernels K1(reduce)->intermediate->K2(map), executes natively, matches tinygrad). **Remaining:** CompiledRunner/BufferCopy abstraction, var binding, JIT.
- ⛔ `engine/jit.py` (312) — TinyJit graph capture/replay (optimization; out of scope for correctness).

## schedule/, ~1100 in-scope
- 🟡 `schedule/__init__.py` (147) + `memory.py` (64) + `rangeify.py` (611) → `schedule.rss`,`schedule_multi.rss`,`buffer_reuse.rss`,`rangeify.rss` — kernel-shape dispatch, reduce-split, liveness buffer reuse, **kernel-grouping/fusion pass** (elementwise fuse, reduce/store/shared boundaries; kernel counts match tinygrad). **Remaining:** real index/range arithmetic in `indexing.py` (286), COPY/upload kernels, cost model.
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
