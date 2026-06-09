# tinygrad → rss(frontend) + mc(backend) port — TODO

Target = the conceptual framework relevant to an rss-frontend / mc(C)-backend port
(~10–15k LOC of tinygrad core). The 179k LOC of `runtime/autogen/*` (machine-generated
NVIDIA/AMD/Mesa GPU register tables) and other-GPU runtime drivers are **out of scope** —
they are not hand-portable to a C backend.

Rule (per user): **if rss/mc lacks a feature needed for a faithful port, fix rss/mc first,
then continue.** The port is "done" when every box below is checked.

## rss/mc fixes (do these as they block the port)
- [x] rss: Hashable/Eq for user types (Map/Set keys)            — earlier (subagent)
- [x] rss parser: parenthesized arithmetic `(a+b)*c`            — f49ac9a, verified green
- [x] rss codegen: read-param double-borrow `&&T`               — 1df5c48 + 21 snapshots f837bf8, verified green
- [x] mc: hosted I/O profile; negative float literals; float-array arithmetic; exp/log/tanh intrinsics
- [ ] rss: `Int.to_float` / `Float.from_int` conversion         — FIX NEXT (blocks numeric interp)
- [ ] rss: `let mut x = <read-param>` clone-on-bind             — codegen bug, scoped fix
- [ ] rss: managed-field clone (callable `.clone()` / take-rebuild) — blocks generic vec()/PtrDType
- [ ] rss: `==` on a `read` enum param (auto-deref)             — ergonomic
- [ ] rss: `List.get` element-type inference on struct-field lists — type-inference gap

## Port items
- [x] dtype.py — scalars, count-aware itemsize, predicates, can_lossless_cast, promo lattice, least_upper_dtype, **vec()**  (`dtype.rss`)
- [ ] dtype.py — PtrDType / ImageDType  (blocked on managed-field clone fix)
- [x] helpers.py — prod/ceildiv/round_up/all_same/dedup/argsort  (`helpers.rss`)
- [x] uop/ops.py — UOp node + structural interning  (`uop.rss`)
- [x] uop/ops.py — PatternMatcher bottom-up symbolic rewrite  (`rewrite.rss`)
- [x] uop/ops.py — generic UPat DSL (wildcards + capture)  (`upat.rss`)
- [x] uop/ops.py — symbolic bounds vmin/vmax  (`symbolic.rss`)
- [x] uop/ops.py — toposort with shared-subgraph dedup  (`toposort.rss`)
- [x] uop/ops.py — complete 93-opcode Ops enum + GroupOp groups  (`ops_enum.rss`)
- [x] shape/View + ShapeTracker — strides + permute/expand/expr/contiguous  (`shapetracker.rss`)
- [x] tensor.py — lazy UOp-graph wrapper  (`tensor.rss`)
- [x] tensor.py — movement + reduce ops with shape tracking  (`tensor_movement.rss`)
- [x] gradient.py — reverse-mode autodiff (tree)  (`gradient.rss`)
- [x] gradient.py — DAG-shared accumulation  (`autodiff_dag.rss`)
- [x] gradient.py — relu vjp  (`gradient_relu.rss`)
- [x] gradient.py — div/neg/exp/log vjp  (`autodiff_ops.rss`)
- [x] codegen/renderer — elementwise + reduce + matmul + relu + div/neg/max/exp/log/tanh, executes via mc  (`render*.rss`)
- [x] schedule — single-kernel dispatch + multi-kernel split  (`schedule*.rss`)
- [x] scheduler — liveness-based buffer reuse  (`buffer_reuse.rss`)
- [x] device.py/runtime — Device[name] dispatch + CPU-interpreter backend  (`device.rss`)
- [x] capstone — SGD training loop to convergence  (`train.rss`)
- [ ] renderer/cstyle.py — faithful UOp-graph → mc kernel renderer  (replaces ad-hoc emitters)
- [ ] engine/schedule.py + realize.py — graph → kernels → run, end-to-end through one ported pipeline
- [ ] final completeness ledger in PORT_STATUS.md

When all boxes are checked → **port done**.
