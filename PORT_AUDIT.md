# tinygrad-rsmc Port Audit

Current source of truth:
- tinygrad source: `/home/zoe/tinygrad/tinygrad`
- integrated RSS package: `tinygrad-rss/`
- standalone translated slices: `port-rss/`
- vendored generated tinygrad data: `tinygrad-rss/vendor/tinygrad/runtime/autogen/`
- compiler/runtime deps: `/home/zoe/rsscript`, `/home/zoe/modern-c`

This audit intentionally does not count `engine/`, `frontend-rss/`, or old architecture docs as
the real port. `engine/` was a misleading verifier/demo tree and has been removed.

## Evidence Checked

- `tinygrad-rss` runs today:
  - command: `RSSCRIPT_RUNTIME_PATH=/home/zoe/rsscript/crates/runtime /home/zoe/rsscript/target/release/rss run tinygrad-rss`
  - output: `linear`, `14`, `25`
- every retained standalone `port-rss/*.rss` file runs today:
  - 55/55 RSS files passed with a 25s per-file timeout. The removed MNIST demo also passed,
    but was deleted because it was not a source-shaped tinygrad port.
- source size inventory:
  - tinygrad handwritten Python, excluding `runtime/autogen`: 118 files, about 33k LOC.
  - tinygrad `runtime/autogen`: 88 generated files, about 179k LOC.
  - integrated `tinygrad-rss/src`: 24 RSS files, 18,691 LOC.
  - vendored `tinygrad-rss/vendor/tinygrad/runtime/autogen`: 88 generated Python files,
    exactly copied from upstream tinygrad commit `fa400f9790ab9a684387b02e958658217b33e7c1`.
  - standalone `port-rss`: 55 RSS files, about 12.1k LOC.

Toolchain changes made to simplify the next port slices:
- RSScript now exposes `Math.cos`, `Math.exp`, `Math.log`, and `Math.tanh` in the stdlib,
  Rust runtime ABI, and reg VM, matching the math surface already needed by tinygrad unary ops.
- RSScript Rust lowering now correctly parenthesizes fallback `read` borrows for inline
  expressions, so calls such as `List.push(..., value: read (0 - 1))` compile without a
  temporary local.
- Modern-C now accepts float bitcasts (`f32`/`f64` to and from scalar bit types) in sema and MIR,
  and its C emitter can lower expression, local, return, assignment, and call-argument bitcasts
  through `__builtin_memcpy` without strict-aliasing casts.
- Modern-C has `std/vec.mc`, a fixed-lane `f32x4` helper surface over arrays for generated kernels
  (`load`, `store`, `add`, `mul`, `max`, `sum`, and lane-wise bit reinterpretation). This is a
  scalar C-emitted abstraction today, not target SIMD.

## What Is Actually Integrated

`tinygrad-rss/` is the only integrated package. It now uses source-aligned folders where RSS can:
top-level `dtype.rss`, `helpers.rss`, `tensor.rss`; `uop/*`; and `runtime/*`. RSS still has a
single global symbol space, so the folders are for auditability and source mapping, not namespaces.
It currently contains a small working graph path, not a user-facing tinygrad API.

Integrated pieces:
- `dtype.rss`: scalar dtype record/registry, fp8 variants, vector dtype semantics, pointer dtype
  shell, predicates, promotion lattice, `least_upper_dtype`, `can_lossless_cast`, and
  `sum_acc_dtype`. Verified against real tinygrad for representative scalar/vector/promotion
  cases, including `bool.vec(4).itemsize == 1`.
- `helpers.rss`: math/list helpers including Python floor division/modulo behavior, `prod`,
  `ceildiv`, `round_up/down`, `next_power2`, cstyle div/mod, `dedup`, `argsort`, `flatten`,
  `get_contraction`, `make_tuple`, `partition`, and `strides_for_shape`.
- `uop/__init__.rss`: operation enum subset/grouping, matching upstream `tinygrad/uop/__init__.py`.
- `uop/ops.rss`: interned UOp DAG cache with constants, symbolic variables (`DEFINE_VAR`/`BIND`),
  source-aligned `RANGE`/`AxisType` metadata, `ParamArg`/addrspace metadata for `PARAM`, `DEFINE_LOCAL`, and `DEFINE_REG`,
  `UNROLL`/`CONTRACT` axis-size pair metadata with vector dtype preservation,
  upstream-shaped `DEVICE`, `UNIQUE`/`LUNIQUE`, `BUFFER`, `COPY`, `ALLREDUCE`, `MULTI`, `MSELECT`, `MSTACK`,
  `CONTIGUOUS`/`CONTIGUOUS_BACKWARD`/`DETACH`, `CUSTOM_FUNCTION`, plus source-shaped
  `STAGE`, `SLICE`, `PROGRAM`, `CALL`, `FUNCTION`, `LINEAR`, `SOURCE`, and `BINARY` metadata,
  with `PROGRAM` now carrying the first integer `Estimates` metadata counters (`ops`, `lds`, `mem`)
  plus runtime-facing `ProgramInfo` helpers for launch dims, runtime-variable indices, and variable values,
  legacy materialized buffers, movement, reduce, and data tensors, plus full-dtype-aware `CAST`,
  same-itemsize full-dtype-aware `BITCAST`, `TUPLE`/`GETTUPLE` including
  upstream's `GETTUPLE(FUNCTION, idx)` tuple-body access, `GROUP`, `GEP`,
  `INDEX`, `LOAD`/`STORE`, codegen helper nodes (`CUSTOM`/`CUSTOMI`, `INS`, `GETADDR`,
  `WMMA`/`SHAPED_WMMA`, `VCAT`/`PTRCAT`), and ordering nodes (`WAIT`, `AFTER`, `END`, `BARRIER`), matching the role of upstream
  `tinygrad/uop/ops.py`. REDUCE metadata is
  explicit (`sum`/`prod`/`max` kind plus axes) rather than encoded as sentinel axes.
- `uop/methods.rss`: integrated UOp helper surface over interned node ids: `const_like`, `ufix`,
  masked-index `invalid`/`valid`/`get_idx`/`get_valid`,
  symbolic variable `expr`/`unbind`/`val`/sorted `variables`/`unbind_all`,
  range string/axis helpers, upstream-style `axis` propagation for sharded `PARAM`/`MULTI`/`GETTUPLE`/ALU/reduce/permute,
  canonical device key/count helpers for `PARAM`/`DEVICE`/`STAGE`/`BUFFER`/`COPY`/`ALLREDUCE`/`MSELECT`/`MSTACK`,
  source-aligned `addrspace` key propagation for `PARAM`/`BUFFER`/`DEFINE_LOCAL`/`DEFINE_REG`/`LOAD`,
  pointer/movement propagation, and all-same `STACK`/elementwise/`WMMA` merging,
  param/addrspace display helpers, source-aligned vector-dtype `broadcast`/`STACK`, `gettuple`, `split_uop`, movement `base`,
  `multibase`, scheduler-facing `buf_uop` projection through movement/`AFTER`/`MSELECT`/`MSTACK`,
  source-aligned `has_buffer_identity`, source-aligned `contiguous` no-op behavior and `bufferize`
  as `STAGE`, concrete `as_shape`,
  movement-argument extraction, replace-by-arg/dtype, structural key rendering, substitute,
  toposort/backward-slice helpers including upstream `enter_calls=false` body-skipping for `CALL`/`FUNCTION`, shared-parent counting, and source-aligned `ranges`/`ended_ranges` propagation for
  range-carrying and range-ending UOps.
- `uop/upat.rss`: integrated UPat/PatternMatcher over real interned node ids, with named captures,
  capture consistency, op sets, dtype constraints, variadic `allow_any_len`, rules-as-data, and a
  generic bottom-up rewrite/fixpoint driver.
- `uop/spec.rss`: first integrated interned-graph verifier for the shared core currently built by
  the package: CONST/SPECIAL/RANGE, `DEFINE_VAR`/`BIND`, `PARAM`, `DEFINE_LOCAL`/`DEFINE_REG`,
  upstream-shaped device/buffer/copy/multi-device graph nodes, ALU dtype rules, same-itemsize `BITCAST`, WHERE/MULACC, buffers, movement
  nodes, STACK/SINK/MSTACK, source-shaped program/call/stage/slice/codegen-facing nodes,
  codegen helper `CUSTOM`/`CUSTOMI`, `INS`, `GETADDR`, `WMMA`/`SHAPED_WMMA`, and `VCAT`/`PTRCAT` nodes,
  `UNROLL`/`CONTRACT` dtype-count/product validation, tuple/gettuple, index/load-store,
  control-flow `IF`/`ENDIF`, order nodes including `END` range validation, and recursive source validation.
- `uop/validate.rss`: first integrated source-shaped `tinygrad/uop/validate.py` slice for masked
  index bounds validation over the interval subset already covered by `uop_min_max`: static false
  gates are accepted, active/unknown gates require the index interval to be fully inside
  `[0, size)`, and the local `validate_index_with_z3` entry point aliases this conservative
  interval proof. Full Z3-backed boolean/arithmetic solving remains unported.
- `uop/render.rss`: first integrated source-shaped `tinygrad/uop/render.py` slice over interned
  UOp ids, covering the core symbolic renderer (`DEFINE_VAR`, named `SPECIAL`, `PARAM`, `RANGE`, `CONST`,
  `CAST`, `BIND`, unary/binary/ternary ALU render rules, `INDEX`/`STAGE`, and `STACK`),
  inference-specific rendering for div/mod and `BITCAST`, plus compact UOp line printing.
- `uop/divandmod.rss`: first integrated source-shaped `tinygrad/uop/divandmod.py` slice over
  interned UOp ids, covering interval cancellation, nested `(x%(k*c))//c` and `%c` rewrites,
  binary numerator folding, gcd-with-remainder factoring, and factor-remainder splitting for
  non-negative integer index expressions. The upstream remove-nested-mod-in-sum rule is held back
  pending a smaller generated-runtime reproducer.
- `uop/decompositions.rss`: first integrated source-shaped `tinygrad/uop/decompositions.py` late
  rewrite slice over interned UOp ids, covering Python `FLOORDIV`/`FLOORMOD` lowering to truncating
  `CDIV`/`CMOD` with mixed-sign adjustment, `MAX` lowering to `CMPLT`+`WHERE`, and
  `RECIPROCAL`/multiply-by-reciprocal lowering to `FDIV`, plus power-of-two floor-mod to `AND`,
  power-of-two multiply/divide to shifts, `x*-1` to `NEG`, `x+(-y)` to `SUB`, and
  multiply/shift-plus-add to `MULACC`, non-negative `CMOD` to `x-d*CDIV(x,d)`, selected signed
  comparison canonicalizations, tight integer range to `CMPEQ`, and logical-not-of-`CMPNE` to
  `CMPEQ`. The large transcendental, long-integer, dtype, fast-idiv, broader boolean/comparison
  normalization, and target-op-availability rewrite machinery remains unported.
- `renderer/cstyle.rss`: first integrated source-shaped `tinygrad/renderer/cstyle.py` slice over
  interned UOp ids, covering C-style dtype names, constants, casts, `__builtin_bit_cast`, core
  unary/binary/ternary ALU rendering, pointer-style `INDEX`, `LOAD`, `STORE`, simple `RANGE`/`END`
  statements, custom text passthrough, `_render`-style name assignment over topologically sorted
  UOps, buffer and scalar `DEFINE_VAR` parameter collection with shaped buffer names such as
  `data0_8`, scoped loop/body emission, store-pointer writable-parameter discovery, and a first
  `render_kernel`-style C function wrapper. It now also renders `IF`/`ENDIF` blocks and aliases
  `AFTER` nodes to their ordered source, plus `DEFINE_LOCAL`/`DEFINE_REG` declarations with
  shape-derived array extents such as `float temp0[4];` and `float acc0[4];`. The kernel wrapper
  can now render either a `SINK` toposort or an explicit `LINEAR` op stream, and it preserves a
  shared loop body across duplicated `END` nodes so generated multi-store kernels stay in scope.
  It still lacks upstream's full linearizer ordering, inline heuristics,
  vector typedefs, target-specific mutable/read-only parameter effects, target subclasses, and schedule integration.
- `codegen/late/linearizer.rss`: first integrated source-shaped `tinygrad/codegen/late/linearizer.py`
  slice over interned UOp ids, covering the priority toposort used by `linearize(sink)`: run-count
  scoring from active ranges, upstream op priorities for `PARAM`/`DEFINE_VAR`/`DEFINE_REG`/
  `DEFINE_LOCAL`/`LOAD`/`STORE`/`RANGE`/`END`, ideal-key ranking, out-degree scheduling from the
  sink, dependency-order verification, compact op-order rendering, and `pm_split_ends`-style
  splitting of multi-range `END` nodes into nested single-range `END`s. It also has the first
  `pm_linearize_cleanups`-style line rewrite: gated `STORE(ptr,value,gate)` becomes
  `IF(gate,ptr)`, ungated `STORE`, `ENDIF`, with later line sources remapped to the ungated store.
  It now also ports the first `CFGContext`/`pm_add_control_flow` equivalent: sibling `END`
  scopes are grouped by nesting owner and dependent `RANGE` nodes receive an extra ordering
  source edge before linearization.
  The first `do_linearize` wrapper now appends a cleaned `LINEAR` UOp to a `PROGRAM(SINK, DEVICE)`
  and is idempotent when the program already has a `LINEAR` child.
  Pre-existing IF rejection, ISA register allocation, compile/binary, and full schedule/device
  rewrite orchestration remain unported.
- `codegen/__init__.rss`: first integrated source-shaped `tinygrad/codegen/__init__.py` slice,
  wiring `PROGRAM(SINK, DEVICE, LINEAR)` through the first `do_estimates` equivalent and CStyle
  renderer. It computes integer upper-bound `Estimates.from_uops(..., ignore_indexing=True)`-style
  metadata for the current interned UOps (`ops`, load/store bytes, and unique capped buffer memory),
  derives first `ProgramInfo.from_sink`-style program metadata (`vars`, `globals`, `outs`, `ins`,
  and integer `SPECIAL` launch dimensions for `g*`, `l*`, and `i*` names) for the current lowered
  sink shape, then appends a `SOURCE` child with the rendered kernel text. It also has the first
  host compile/syntax bridge for rendered CStyle source, writing the attached `SOURCE` to a path
  and invoking `clang -x c -std=c11 -fsyntax-only` through RSScript `Path`/`Process`, returning
  source id/length, exit status, stderr length, and ok/fail metadata. It also has the first hosted
  executable C harness bridge for the current generated kernel shape: combine `SOURCE` with a
  supplied C `main`, compile with `clang`, run the executable through `Process`, and capture stdout
  plus compile/run status. The first `to_program`-style orchestration for the supported C-style path is:
  `do_linearize -> do_estimates -> do_render_cstyle`. Full symbolic `sint` estimates,
  aux metadata, compiled binary materialization, general buffer-backed runtime invocation, and full
  schedule/device rewrite orchestration remain unported.
- `shape.rss`: UOp shape inference helpers, including upstream-shaped buffer size shape, `PARAM`/
  `DEFINE_LOCAL`/`DEFINE_REG` shape payloads, `BINARY` byte length, `STACK`/`GEP`, `GETTUPLE`
  tuple-element shape propagation through `TUPLE` and `FUNCTION`, vector-shaped scalar/control helper nodes, broadcasting, and View/ShapeTracker core for
  contiguous views, permute, flip, expand, pad, shrink, flat-index expression, and contiguity
  checks.
- `device.rss`: first integrated source-shaped `tinygrad/device.py` metadata and allocator slice,
  covering canonical device strings, `BufferSpec`, `Buffer`/`MultiBuffer` descriptors, nbytes,
  allocation-state projection, refcounts, view offsets, explicit RSS byte-list allocation handles,
  copyin/copyout, deallocate, cache reuse, and cache freeing. The RSS struct is named `TGBuffer`
  to avoid a generated Rust backend collision, while the helper surface remains `buffer_*`.
  Native opaque runtime handles, full device-specific LRU allocator behavior, compiler caches,
  dynamic runtime loading, profiling finalization, and device enumeration remain unported.
- `engine/realize.rss`: first integrated source-shaped `tinygrad/engine/realize.py` metadata and
  byte-buffer execution slice, covering `get_call_arg_uops`, `get_call_outs_ins` for `PROGRAM`,
  `COPY`, `SLICE`, and `CUSTOM_FUNCTION("encdec")`, `estimate_uop` for `PROGRAM`, `COPY`, and
  `encdec`, parameter resolution helpers, RSS byte-store `exec_copy`/`exec_view` over the
  integrated `TGBuffer` allocator, the first `run_linear` dispatcher for supported `COPY` and
  `SLICE` calls, first parameter-slot-to-buffer-index remapping for those calls, and first
  `PROGRAM` execution staging that resolves/allocates declared global buffers and records function
  name, source id/length, launch metadata, declared global slots, and resolved buffer indices.
  It also includes the first per-context runtime cache keyed by function/source metadata, returning
  stable runtime handles with hit/miss reporting. The first hosted CPU numeric program execution
  path is now integrated for supported multi-output/multi-input standard scalar shapes, including
  1/2/4/8-byte C scalar integer and float dtypes; element counts are derived from dtype itemsize:
  raw RSS byte buffers are marshalled into a generated C harness, the rendered or attached
  `SOURCE` is compiled and run through `Path`/`Process`, stdout decimal bytes are parsed and split
  across output buffers, and the output bytes are copied back into the existing `TGBuffer` store.
  `run_linear` now tries this hosted path for supported `PROGRAM` calls and falls back to metadata
  staging for unsupported shapes. Local-size optimization, general compiled numeric kernel invocation,
  validation execution, graph execution, full multi-buffer/device remapping, and dynamic-library
  runtime plumbing remain unported.
- `gradient.rss`: first integrated reverse-mode autodiff slice over interned UOp ids, covering
  target-specific symbolic gradients for `CAST`, `ADD`, `SUB`, `MUL`, `FDIV`, unary
  `NEG`/`RECIPROCAL`/`SQRT`/`EXP2`/`LOG2`/`SIN`/`TRUNC`, binary `POW`, `MAX` with upstream
  tie-splitting, mask-selection `WHERE`, zero gradients for integrated comparison predicates, and
  movement/reduce VJPs for `RESHAPE`, `EXPAND`, `PERMUTE`, `FLIP`, `PAD`, `SHRINK`, and `REDUCE`, then
  simplifying the resulting gradient graph.
- `uop/vminmax.rss`, `uop/alu.rss`, `uop/symbolic.rss`, `runtime/ops_python.rss`: small
  interpreter/simplifier pieces. The symbolic layer now includes integer vmin/vmax, rewrite
  identities/folding, a grouped `symbolic_simple` slice for idempotent ops, boolean constants,
  same-self comparisons, `x//-1`, `(x^y)^y`, boolean not/cast/trunc folding,
  bool-ALU-to-logic folding, boolean contradiction folding, additive coefficient combining,
  constant-times-sum distribution, inverted-condition `WHERE` folding, simple `WHERE` folding,
  constant-exponent `POW` simplification for integer exponents,
  monotonicity checks, known constant-factor extraction, `pop_const`,
  symbolic `gcd`, and proven exact division for constants, stacks, adds, multiplies, and simple
  multiplicative cancellation. Vmin/vmax covers bounded `PARAM`, plus both `SPECIAL` and `RANGE` as `[0, end_max - 1]`.
  `STACK` and `UNROLL` propagate child/source min-max ranges.
  The evaluator now covers the integrated unary math UOps
  `EXP2`, `LOG2`, `SIN`, `SQRT`, `RECIPROCAL`, `NEG`, `TRUNC`, plus binary `POW`, and handles
  scalar-to-tensor `EXPAND` materialization, `FLIP`, `PAD`, graph-backed `SHRINK` slicing, and
  multi-axis `REDUCE` materialization.
- `tensor.rss`: tensor surface over UOp ids with graph-building creation helpers (`full`,
  `zeros`, `ones`, `full_like`, `zeros_like`, `ones_like`, `arange`,
  `arange(start, stop, step)`, `linspace`, `eye`, bit-exact explicit-seed `rand`,
  `rand_like`, `uniform`, `randint`, `randperm`, `randn`, `randn_like`, `normal`,
  `normal_like`, `scaled_uniform`, `glorot_uniform`, `kaiming_uniform`, and
  `kaiming_normal`), dtype conversion through `CAST` for core float/int/bool cases, same-itemsize
  `BITCAST` graph construction plus upstream-style shape-changing bitcast composition through
  unsigned integer packing/unpacking, broadcast elementwise arithmetic/comparison ops, unary
  `EXP2`/`LOG2`/`SIN`/`SQRT`/`RECIPROCAL`/`NEG`/
  `TRUNC`, binary `POW`, composed `rsqrt`, `log10`, `minimum`, `square`, `clip`, `sign`, `abs`,
  `ceil`, `floor`, `sigmoid`, `exp`, `log`, `cos`, `tan`, `tanh`, `leaky_relu`, `quick_gelu`, default
  tanh-approx `gelu`, `swish`/`silu`, `hardswish`, `hardsigmoid`, `softplus`, `mish`,
  `logsigmoid`, `elu`, `celu`, `selu`, `softsign`, `lerp`, `hardtanh`, and `relu6`,
  bitwise/logical `AND`/`OR`/`XOR`, integer shifts, floor/truncating div and modulo variants,
  `WHERE` mask selection, movement helpers (`reshape` with single `-1`
  inference, `view`, `permute`, `flip`, `pad`, `pad_to`, `expand`, `shrink` through explicit
  per-axis `slice`, `shrink_to`, `split`, `chunk`, `meshgrid` for 1D/scalar tensors, and
  `diag` for 1D tensors, `diagonal`, `triu`, `tril`, `repeat_interleave`, `repeat`, `roll`, `unfold`, `cat`,
  `stack`, ADD-backed `cumsum`, MUL-backed `cumprod`, MAX-backed `cummax` values/indices, and
  negated-MAX-backed `cummin` values/indices),
  composed movement helpers (`unsqueeze`, `squeeze(dim)`,
  `squeeze()`, `transpose`, `flatten`, and `unflatten`), reduce helpers (`sum(axis)`, `sum()`,
  `prod(axis)`, `prod()`, `max(axis)`, `max()`, `min(axis)`, `min()`, bool `any`/`all`,
  `mean(axis)`, `mean()`,
  axis/all `var` and `std`, tuple-axis `mean`/`var`, single-axis `normalize`, single-axis and
  tuple-axis `layernorm`, channel-axis `batchnorm` with no-affine and affine forms, plus
  keepdim-backed `logsumexp(axis)`, `softmax(axis)`, `log_softmax(axis)`, first-index
  `argmax`/`argmin` for all/axis reductions, and graph-composed stable `sort`, `argsort`,
  and sorted `topk` for fixed-shape tensors), `isnan`/`isinf`/`isfinite` predicates plus
  `isclose`/`allclose` with `equal_nan` handling, loss helpers (`binary_crossentropy`,
  `binary_crossentropy_logits`, `nll_loss`, and sparse-target `cross_entropy`), `_pool`-style
  NCHW `avg_pool2d` with count-include/count-exclude padding and `ceil_mode`, `max_pool2d` with
  low constant padding, dilation, `ceil_mode`, padded/dilated/ceil `return_indices`, default and
  explicit-output `max_unpool2d`, `conv2d` with optional bias, zero/signed padding, dilation, and grouped convolution,
  and NCHW `conv_transpose2d` with stride, padding, dilation, output padding, bias, and groups,
  graph-backed `_one_hot_along_dim`, `one_hot`, generic `gather(dim, index)`, plain `scatter`
  with duplicate last-wins masked merge, `scatter_reduce` for sum/prod/mean/amax/amin,
  scalar `scatter(..., reduce="add"/"multiply")`, plus legacy
  materialized gather indexing (`gather_axis0`), upstream-style `dot`/`matmul`/`linear`
  composition for vector, matrix, and batched cases, upstream-style `relu` as
  `(x > 0).where(x, 0)`, and legacy `linear_forward`.

Current integrated demo:
- validates dtype, helpers, view/shape, UOp helpers, UOp spec checks, UPat matching, and
  rules-as-data graph rewrite.
- validates UOp method helpers for `STACK` splitting/gettuple, concrete shape extraction,
  movement base recovery, and movement argument extraction.
- validates vector-dtype `broadcast` and full-dtype-aware `CAST`, including vector-to-scalar
  casts that must not be skipped by scalar dtype-name comparison.
- validates source-aligned masked-index helpers: `valid(idx, cond)` builds a verifier-accepted
  `WHERE`, `get_idx` unwraps the payload, `get_valid` unwraps the mask, and vector-count
  `invalid` preserves weak-index dtype count.
- validates UOp symbolic variables and binding: expression extraction, declared vmin/vmax,
  in-range `BIND`, `unbind`, `val`, sorted `variables`, `unbind_all` rewrite with collected
  bindings, and out-of-range bind rejection.
- validates the integrated source-shaped UOp renderer for precedence-aware symbolic expressions,
  `max`, inference `floordiv`, inference `BITCAST`, compact repeated `STACK`, and UOp line output.
- validates UOp `ParamArg` metadata and definitions: `PARAM` slot/name/addrspace, bounded param
  vmin/vmax, `DEFINE_LOCAL`/`DEFINE_REG` addrspace, and rejection of invalid param bounds or
  negative local slots.
- validates upstream-shaped UOp device/buffer graph metadata: `DEVICE` tuple key/count,
  `UNIQUE`-backed `BUFFER` shape/device count, `COPY`, `MULTI`, `MSELECT`, `MSTACK`, `ALLREDUCE`,
  `CONTIGUOUS`, `DETACH`, `CUSTOM_FUNCTION`, and rejection of empty devices, out-of-range
  `MSELECT`, and invalid `ALLREDUCE` ops.
- validates source-aligned UOp `axis` propagation for `MULTI`, `COPY`, sharded `PARAM`, ALU,
  `GETTUPLE`, `REDUCE`, and `PERMUTE`.
- validates source-aligned UOp device key/count propagation for tuple buffers, `COPY`,
  `MSELECT`, `MSTACK`, `STAGE`, and device-carrying `PARAM`.
- validates source-aligned UOp addrspace propagation for `PARAM`, `BUFFER`, `DEFINE_LOCAL`,
  `DEFINE_REG`, `INDEX`, `LOAD`, movement nodes, and all-same/mixed `ADD` and `STACK` sources.
- validates source-aligned UOp `base`/`multibase` distinction for `MULTI` and `buf_uop`
  projection through movement, `AFTER`, `MSELECT`, and `MSTACK`.
- validates source-aligned UOp `has_buffer_identity` through `RESHAPE`, `MULTI`, `SLICE`,
  `GETTUPLE(TUPLE)`, and negative `CONST` cases.
- validates source-aligned UOp `contiguous` behavior: device-carrying non-buffer identity nodes
  become `CONTIGUOUS`, while already-contiguous, buffer-identity, and no-device nodes are no-ops;
  also validates `bufferize` produces verifier-accepted `STAGE`.
- validates UOp `RANGE` construction with axis metadata/string rendering, `[0, end-1]` vmin/vmax,
  verifier acceptance, valid `END(store, range)`, and rejection of `END` over a non-range source.
- validates UOp `ranges`/`ended_ranges` tracking: ALU expressions retain active ranges, `END`
  removes them, `AFTER` propagates ended ranges from ordering dependencies, and `CONTRACT`
  ends matching range ids from axis-size metadata.
- validates source-shaped UOp `STAGE`/`SLICE`/`PROGRAM`/`CALL`/`FUNCTION` construction, shape
  helpers for stage/slice, verifier acceptance, and rejection of malformed program order, call
  bodies, function bodies, and slice bases.
- validates codegen-helper UOps: `MULACC`, `CUSTOM`/`CUSTOMI`, `INS`, `GETADDR`,
  `VCAT`/`PTRCAT`, `WMMA`/`SHAPED_WMMA`, vector-count/WMMA-arg checks, and rejection of
  malformed dtype/arg metadata.
- validates UOp `UNROLL`/`CONTRACT` axis-size product metadata, `CONTRACT` vector dtype count,
  `UNROLL` min/max propagation, verifier acceptance, and rejection of product mismatch/empty axes.
- validates source-aligned UOp constructors/spec checks for `TUPLE`/`GETTUPLE`, `GETTUPLE(FUNCTION, idx)`, tuple/function shape propagation,
  `BINARY`/`STACK`/`GEP`/vector-cast shape helpers, `GROUP`,
  `GEP` stack extraction, `INDEX`, `LOAD`, `STORE`, `IF`/`ENDIF`, `AFTER`, `BARRIER`, and `END`.
- validates UOp symbolic helpers for monotonicity, constant-exponent `POW` simplification
  (`x**3 -> x*x*x`, `x**0 -> 1`, `x**-1 -> 1/x`), constant-factor extraction, exact
  divisibility on `(x*4)+8`, conservative `divide_exact` cancellation for `(x*6)/(x*3)`,
  and symbolic `gcd(x*6, x*9) -> 3*x`.
- validates a grouped upstream `symbolic_simple` batch: `max(x,x)`, boolean `x&x`/`x|x`,
  boolean `x&true`/`x&false`/`x|false`/`x|true`, `x<x`, integer `x!=x`, `(x^y)^y`,
  integer `x&0`, and `x//-1`.
- validates a grouped boolean symbolic batch: `where(b,true,false) -> b`,
  `where(b,false,true) -> !b`, `!!b -> b`, `cast(b,int)!=0 -> b`, `cast(b,int)!=1 -> !b`,
  `cast(b,int)!=2 -> true`, and integer `trunc(x) -> x`.
- validates a grouped upstream symbolic combine/algebra batch: bool `MUL -> AND`, bool
  `ADD/MAX -> OR`, `b|!b -> true`, `x*2+x*3 -> x*5`, `x+x -> x*2`,
  nested additive coefficient combining, `-1*(x+3) -> -x + -3`, and
  `!b.where(2,3) -> b.where(3,2)`.
- validates simple symbolic `WHERE` folding for true, false, same-branch, and bottom-up
  constant-comparison conditions.
- validates integrated div/mod symbolic rewrites against upstream-shaped samples:
  `(n%8)//4 -> (n//4)%2`, `(n%8)%4 -> n%4`, `(b*6+2)//3 -> b*2`,
  `(b*6+2)%3 -> 2`, `(n*4+8)//6 -> ((n*2+1)//3)+1`,
  `(n*4+8)%6 -> ((n*2+1)%3)*2`, and `(n*6+m)//6 -> m//6+n`.
- validates integrated decomposition rewrites for same-sign floor division, mixed-sign floor
  division/modulo, `MAX` to `WHERE`, reciprocal to `FDIV`, and multiply-by-reciprocal to `FDIV`,
  plus power-of-two floor-mod/multiply/divide, `NEG`, `SUB`, and `MULACC` lowering, with verifier
  acceptance for each emitted graph. It also validates non-negative `CMOD` lowering, selected
  signed comparison canonicalizations, tight-range equality, and logical-not-of-`CMPNE` lowering.
- validates the integrated C-style renderer slice for arithmetic expressions, `WHERE`, `MULACC`,
  casts, bitcasts, pointer-style indexed loads, and stores, producing strings such as
  `((dp+3)*4)`, `((dp<4)?dp:3)`, `__builtin_bit_cast(unsigned int, (int)(dp))`,
  `(*(buf+dp))`, and `*(buf+dp) = fa;`.
- validates first integrated C-style kernel rendering over a real UOp graph with `PARAM`, `RANGE`,
  `DEFINE_LOCAL`/`DEFINE_REG`, `INDEX`, `LOAD`, ALU, `STORE`, `END`, and `SINK`, rendering a
  parseable C function for `out[i] = (in[i] + 2.0f) * 3.0f`, including local/register array
  declarations, shaped buffer argument names, a scalar `DEFINE_VAR n` kernel argument used as the
  dynamic loop bound, an `IF`/`ENDIF` block around the store, and writable-parameter detection
  that marks the output buffer writable while leaving the input buffer read-only.
- validates first integrated codegen linearizer priority ordering over the same real kernel graph,
  producing a dependency-valid linear op stream from `CONST`/`PARAM`/`DEFINE_*` through
  `RANGE`/`INDEX`/`LOAD`/ALU/`STORE`/`END`/`IF`/`ENDIF`/`SINK`.
- validates `pm_split_ends`-style cleanup by rewriting a synthetic two-range `END` into nested
  single-range `END` nodes.
- validates `pm_linearize_cleanups`-style gated-store lowering over a real UOp line stream:
  the cleaned stream contains one `IF` and one `ENDIF`, no gated `STORE`, and still satisfies
  dependency ordering.
- validates the first `do_linearize` wrapper by appending a cleaned `LINEAR` child to a
  `PROGRAM(SINK, DEVICE)` and passing the integrated UOp spec verifier.
- validates the first `do_estimates` wrapper by adding `PROGRAM` estimate metadata for the sample
  linear stream: `ops=128`, `lds=512`, and capped unique memory `mem=64`.
- validates the first `do_render`-style wrapper by rendering a cleaned `LINEAR` stream into a
  non-empty `SOURCE` child on `PROGRAM(SINK, DEVICE, LINEAR)`, passing the integrated UOp spec
  verifier, and producing parseable C for the sample gated-store kernel.
- validates `toposort(enter_calls=false)` behavior: call arguments remain visible while
  `CALL`/`FUNCTION` bodies are not traversed.
- validates `backward_slice` and `op_in_backward_slice_with_self` membership checks over a
  simple ALU graph.
- validates core symbolic gradients such as `d(3*x)/dx`, `d(x*x)/dx`, `d(x*x+5*x)/dx`,
  `d(x-3*x)/dx`, and `d(x/y)/dx`.
- validates Tensor creation, broadcasting, movement, reduce, and mean through the integrated
  UOp graph evaluator.
- validates Tensor `eye` creation against real tinygrad for square and rectangular forms.
- validates Tensor range creation against real tinygrad for stepped positive/negative `arange`
  and `linspace`.
- validates Tensor like-constructors against real tinygrad for `full_like`, dtype-overridden
  `zeros_like`, `ones_like`, and explicit-seed `rand_like`/`randn_like` shape reuse.
- validates `Tensor.rand(6)`, `uniform(2,3, low=2, high=10)`,
  `randint(2,3, low=5, high=10)`, and `randperm(6)` with seed 42 against real tinygrad's
  Threefry path.
- validates explicit-seed initializer helpers against real tinygrad with seed 42: `randn(2,3)`,
  `normal(2,3, mean=10, std=2)`, `scaled_uniform(2,3)`, `glorot_uniform(2,3)`,
  `kaiming_uniform(2,3)`, and `kaiming_normal(2,3)`.
- validates basic Tensor indexing against real tinygrad for `x[Tensor([1,0])]` and
  graph-backed `SHRINK` slicing for `x[0:2,1:3]` on a `[2,3]` tensor, including spec validation.
- validates graph-backed Tensor `one_hot(5)`, `gather(dim=1, index)`, and `gather(dim=0, index)`
  against real tinygrad using the upstream one-hot mask plus reduction composition.
- validates graph-backed Tensor `scatter_reduce` against real tinygrad for `sum`, `prod`,
  `mean(include_self=False)`, `amax`, and `amin`, plus scalar
  `scatter(..., reduce="multiply")` and `scatter(..., reduce="add")`.
- validates graph-backed plain Tensor `scatter` against real tinygrad for `dim=0`, `dim=1`, and
  duplicate indices where the last source value wins.
- validates graph-backed `FLIP` against real tinygrad for axis-1 reversal on a `[2,3]` tensor.
- validates graph-backed `PAD` against real tinygrad for `pad(((1,0),(1,2)))` on a `[2,3]`
  tensor, including spec validation.
- validates composed Tensor movement helpers against real tinygrad: `unsqueeze(1)`,
  `squeeze(1)`, `squeeze()`, `flatten(1,2)`, `flatten()`, `transpose(0,2)`, `view(6,4)`,
  `unflatten(0,(2,-1,4))`, `shrink_to((2,2))`, and `pad_to((3,5))`.
- validates Tensor `split` and `chunk` against real tinygrad: `split(2, dim=0)`,
  `split([1,4], dim=0)`, and `chunk(3, dim=0)` on a `[5,2]` tensor, all backed by `SHRINK`
  nodes.
- validates Tensor `meshgrid` against real tinygrad for two 1D tensors with both `ij` and `xy`
  indexing, backed by `RESHAPE` and `EXPAND`.
- validates Tensor `diag` against real tinygrad for a 1D `[1,2,3,4]` tensor, backed by composed
  movement nodes rather than materialized data.
- validates Tensor `diagonal` against real tinygrad for main, positive-offset, negative-offset,
  and rectangular 2D cases, backed by composed movement nodes.
- validates Tensor `triu` and `tril` against real tinygrad on rectangular 2D tensors for main,
  positive, and negative diagonal offsets.
- validates Tensor `repeat_interleave`, `repeat`, and `roll` against real tinygrad for flat,
  dimension-specific, positive-shift, and negative-shift cases, all backed by movement nodes.
- validates Tensor `unfold` against real tinygrad for 1D windows on a flat tensor and along a
  non-leading axis of a 2D tensor, backed by the same `_pool`-style movement composition.
- validates Tensor pooling/convolution against real tinygrad for no-padding NCHW cases:
  `avg_pool2d(kernel_size=(2,2), stride=(2,2))`, `max_pool2d(kernel_size=(2,2), stride=(2,2))`,
  single-channel `conv2d` with and without bias, and groups=1 multi-channel/multi-output `conv2d`.
- validates padded pooling/convolution against real tinygrad: `avg_pool2d(..., padding=1)`,
  low-fill `max_pool2d(..., padding=1)` on negative inputs, and `conv2d(..., padding=1)` with
  and without bias.
- validates dilated pooling/convolution against real tinygrad: `avg_pool2d(..., dilation=2)`,
  `max_pool2d(..., dilation=2)`, `conv2d(..., dilation=2)` with and without bias, and
  `conv2d(..., padding=1, dilation=2)`.
- validates `avg_pool2d(..., count_include_pad=False)` against real tinygrad for padded and
  padded+dilated cases.
- validates pooling `ceil_mode=True` against real tinygrad for `avg_pool2d`, padded
  `avg_pool2d(..., count_include_pad=False)`, and `max_pool2d`.
- validates unpadded NCHW `max_pool2d(..., return_indices=True)` against real tinygrad, including
  first-index behavior for tied maxima.
- validates padded, padded+dilated, and ceil-mode NCHW `max_pool2d(..., return_indices=True)`
  against tinygrad's source semantics, where returned indices are flat positions in the original
  unpadded input spatial shape.
- validates default NCHW `max_unpool2d` against real tinygrad for ordinary and tied-window
  `max_pool2d(..., return_indices=True)` inputs.
- validates padded NCHW `max_unpool2d` with default inferred shape and explicit `output_size`.
- validates grouped Tensor `conv2d` against real tinygrad for `groups=2`, including bias and
  padded+dilated grouped convolution.
- validates Tensor `conv_transpose2d` against real tinygrad for default, stride+padding+output-padding,
  dilation+padding+bias, and `groups=2` cases using the upstream zero-insert, weight flip/swap,
  and derived-padding composition.
- validates Tensor `cat` and `stack` against real tinygrad for dim-0 and dim-1 cases, backed by
  upstream-style `PAD` plus sum and `unsqueeze` plus `cat` compositions.
- validates ordinary Tensor reductions against real tinygrad: `sum(axis=0)`, `sum()`,
  `prod(axis=1)`, `prod()`, `max(axis=0)`, `max()`, `min(axis=1)`, `min()`, `mean(axis=1)`,
  and `mean()`.
- validates bool Tensor reductions against real tinygrad: `all()`, `any()`, `all(axis=1)`, and
  `any(axis=0)`.
- validates Tensor `argmax`/`argmin` against real tinygrad for all, axis-0, and axis-1 cases,
  including first-index behavior for tied maxima.
- validates Tensor `sort`, `argsort`, and sorted `topk` against real tinygrad for axis-0 and
  axis-1 cases, including stable duplicate-index reconstruction.
- validates Tensor `isclose` and `allclose` for loose/tight finite tolerances and NaN/Inf
  edge cases including `equal_nan=True`.
- validates Tensor loss helpers against real tinygrad: binary cross entropy mean/sum, BCE logits
  mean with and without `pos_weight`, NLL loss none/mean, and sparse-target cross entropy none/mean.
- validates Tensor statistics against real tinygrad: `var(axis=1, correction=0/1)`,
  `std(axis=1, correction=0/1)`, `var(correction=0)`, and `std(correction=0)`.
- validates Tensor normalization helpers against real tinygrad: `normalize(p=2, dim=1)`,
  `normalize(p=1, dim=0)`, `normalize(p=0, dim=1)`, and `layernorm(axis=1, eps=1e-5)`.
- validates tuple-axis Tensor statistics/normalization against real tinygrad: `mean(axis=(1,2))`,
  `var(axis=(1,2), correction=0)`, and `layernorm(axis=(1,2), eps=1e-5)`.
- validates Tensor `batchnorm` against real tinygrad for channel axis 1, both no-affine and affine
  `weight`/`bias` forms.
- validates keepdim-backed reduction compositions against real tinygrad: `logsumexp(axis=1)`,
  `softmax(axis=1)`, and `log_softmax(axis=1)`.
- validates Tensor matrix APIs against real tinygrad: 1D dot, 2D dot, matrix-vector dot,
  vector-matrix dot, batched matmul, and linear with bias.
- validates Tensor `cumsum` against real tinygrad for axis-0 and axis-1 cases, backed by the
  upstream `_cumalu` ADD composition using zero-padding, pooling, and sum.
- validates Tensor `cumprod` against real tinygrad for axis-0 and axis-1 cases, backed by the
  upstream `_cumalu` MUL composition using one-padding, pooling, and product reduction.
- validates Tensor `cummax` against real tinygrad for 1D axis-0 and 2D axis-1 cases, including
  returned values and indices, backed by the upstream `_cumalu` MAX composition plus match/index
  reconstruction.
- validates Tensor `cummin` for the same 1D axis-0 and 2D axis-1 cases, including returned
  values and indices, backed by negating through the `cummax` values/index reconstruction.
- validates representative Tensor casts against real tinygrad: float-to-int truncates toward zero,
  float-to-bool uses nonzero truthiness, and int-to-float preserves values.
- validates Tensor `BITCAST` graph construction and spec behavior for same-itemsize
  float32-to-uint32 and shape-changing widen/narrow paths such as float32[4]->uint64[2] and
  uint64[2]->uint8[16]. Exact bit reinterpretation is validated in the Modern-C backend path,
  not the RSS list evaluator.
- validates the MC backend bitcast path with a native roundtrip:
  `x.bitcast(uint32).bitcast(float32) + 1` renders through Modern-C, compiles with clang, and
  matches tinygrad CPU output.
- validates Tensor comparisons and mask selection against real tinygrad: `<`, `ne`, `eq`, and
  `where(mask, x, 0)` with scalar broadcasting.
- validates representative unary/binary math Tensor ops against real tinygrad: `neg`,
  `reciprocal`, `sqrt`, `rsqrt`, `log2`, `log10`, `exp2`, zero-input `sin`, `trunc`, and `pow` on exactly
  representable or precision-stable values.
- validates composed Tensor elementwise helpers against real tinygrad: `sign`, `abs`, `square`,
  `minimum`, `clip`, `ceil`, `floor`, `sigmoid`, `exp`, `log`, `cos`, `tan`, `tanh`,
  `leaky_relu`, `quick_gelu`, default tanh-approx `gelu`, `swish`/`silu`, `hardswish`,
  `hardsigmoid`, `softplus`, `mish`, `logsigmoid`, `elu`, `celu`, `selu`, `softsign`, `lerp`,
  `hardtanh`, `relu6`, and upstream-style `relu`, including positive, negative, and zero values.
- validates bool and int bitwise/logical Tensor ops against real tinygrad: `AND`, `OR`, and `XOR`.
- validates integer shift and division/modulo Tensor ops against real tinygrad: `SHL`, `SHR`,
  floor division/modulo, and truncating cstyle division/modulo.
- validates Tensor gradients through broadcast and sum: gradient wrt a `[2,3]` input is all ones,
  and gradient wrt a broadcasted `[1,3]` input accumulates to `[2,2,2]`.
- validates Tensor gradients through graph-backed `SHRINK` slicing against real tinygrad:
  `sum(x[0:2,1:3])` routes gradients back to the source as `[0,1,1,0,1,1]` using a `PAD` VJP.
- validates Tensor gradients through graph-backed `FLIP` against real tinygrad with a weighted
  sum, routing gradients back to the source as `[100,10,1,200,20,2]`.
- validates Tensor gradients through graph-backed `PAD` against real tinygrad:
  `sum(x.pad(((1,0),(1,2))))` routes an all-ones gradient back to the unpadded source.
- validates Tensor gradients through integrated casts against real tinygrad: float input through
  `float->int->float` and `float->bool->float` both pass an all-ones adjoint back to the float
  input, matching upstream `CAST` VJP semantics.
- validates Tensor gradients through mask selection against real tinygrad:
  `sum(where(mask, x, y))` routes gradients to the true and false branches as `[1,0,1,0]` and
  `[0,1,0,1]`.
- validates Tensor gradients through `maximum` against real tinygrad, including upstream tie
  splitting: left branch `[0,0.5,1,0.5]`, right branch `[1,0.5,0,0.5]`.
- validates Tensor gradients for integrated unary/math ops against real tinygrad on stable inputs:
  `sum(neg(u))`, `sum(reciprocal(u))`, `sum(sqrt(u))`, `sum(sin(0))`, `sum(trunc(t))`, and
  `sum(u**2)`.
- builds `relu(x @ W + b)` through the cache and evaluates `[14, 25]`.

This is useful, but it is not yet a 1:1 tinygrad port.

## What Exists As Standalone Port Slices

`port-rss/` contains source-shaped translated slices and validation programs. These all run today,
but most are not wired into `tinygrad-rss/`.

Useful translated/validated areas:
- dtype: scalar/vec/ptr pieces, promotion lattice, `can_lossless_cast`.
- helpers: `prod`, `ceildiv`, `round_up`, `all_same`, `dedup`, `argsort`, more helper utilities.
- uop core: UOp, op enum, interning, UPat, pattern matching, graph rewrite, toposort, method surface.
- symbolic/simplify: vmin/vmax, symbolic rules, div/mod/decompositions, deeper simplification.
- tensor slices: core lazy wrapper, elementwise ops, movement, higher ops, conv/pool, RNG/indexing.
- autodiff/gradient: reverse-mode rules and DAG accumulation slices.
- codegen/render/schedule: MC rendering, reductions, matmul, relu, more cstyle renderer, linearizer,
  late codegen, rangeify/indexing/buffer reuse, realize/schedule demos.
- device/runtime slices: CPU-style interpreter/device/buffer sketches.
- nn slices: layers, conv, optimizer/state-dict related pieces.
- capstones: small training demos that validate mechanisms but are not tinygrad source files.

These files are valuable migration material. They should be pulled into `tinygrad-rss/` module by
module, with names and APIs aligned to tinygrad source, then tested from one package.

## Generated Runtime Autogen

`tinygrad/runtime/autogen/*` should not be hand-ported.

It is generated hardware/API binding data. The generated Python sources are now vendored at
`tinygrad-rss/vendor/tinygrad/runtime/autogen/`, outside RSS package sources so the RSS loader does
not compile them. The provenance note is `tinygrad-rss/vendor/tinygrad/runtime/README.md`.

Autogen is tracked as generated data, not counted as handwritten port debt. Remaining runtime work
is the handwritten code that imports/uses this data, plus any generator-preservation work needed for
future refreshes.

## What Still Needs Porting

Major missing integrated work:
- `tensor.py` and `mixin/*`: broad public Tensor API. Creation, elementwise, movement, reduce,
  explicit-seed random and initializer helpers are partially integrated now, but exact global
  seed/counter APIs, broader distribution helpers,
  full symbolic/lazy indexing semantics, graph-backed NaN/Inf predicate lowering, broader composed transcendental/math method coverage,
  full conv/pool semantics including more negative/asymmetric edge cases and broader dimensionality,
  Python-facing method breadth,
  and exact tinygrad semantics are still incomplete.
- `function.py` and `gradient.py`: autograd is partially integrated for scalar symbolic UOp
  graphs, simple Tensor movement/reduce graphs including slice/pad, mask selection, max, and selected unary/math VJPs, but full
  Function-style APIs, complete VJP coverage, gradient accumulation APIs, and exact tinygrad
  gradient semantics are still missing.
- `uop/ops.py`, `uop/upat.py`, `uop/symbolic.py`, `uop/divandmod.py`, `uop/decompositions.py`, `uop/render.py`, `uop/spec.py`, `uop/validate.py`: full
  source-aligned implementation. UPat/rewrite and a method-helper slice are integrated now, but
  the full upstream method surface, symbolic/decomposition/divmod coverage, complete render/pyrender
  behavior, full spec validation, full Z3-backed bounds validation, and exact upstream semantics are still incomplete.
- `schedule/*`: source-shaped scheduler, memory planner, rangeify/indexing, multi-kernel behavior.
- `codegen/*`: full lowerer and late passes aligned to tinygrad. The first linearizer priority
  toposort, CFG/control-flow insertion, split-end cleanup, gated-store line cleanup,
  `PROGRAM` -> `PROGRAM+LINEAR` wrapper, integer upper-bound `do_estimates` metadata,
  first `ProgramInfo.from_sink` metadata derivation including integer `SPECIAL` launch dimensions, and
  `PROGRAM+LINEAR` -> `PROGRAM+LINEAR+SOURCE` CStyle wrapper are integrated, but pre-existing
  IF rejection, late expansion/devectorization, range simplification, GPU dims, ISA/regalloc,
  full symbolic estimates, compile/binary, and clear scope for GPU-only optimization passes remain.
- `renderer/cstyle.py`: full target-specific MC/C-style kernel renderer on top of the first
  generic kernel wrapper now in `renderer/cstyle.rss`.
- `engine/*`, `device.py` beyond the first allocator/metadata slice, `runtime/*`, `runtime/support/*`:
  handwritten execution, device/runtime/compiler layers.
  The generated `runtime/autogen` data has been copied, but handwritten behavior must still be
  ported and accounted for.
- `nn/*`: layers, optimizers, state, datasets where in scope. `nn/onnx.py` may be scoped as a
  separate importer rather than core tensor correctness.
- examples such as `beautiful_mnist.py`: only after Tensor, nn, datasets, optimizer, scheduler,
  and backend execution are integrated enough to run the real example, not a special demo.

## Cleanup Decisions

Remove misleading or non-port artifacts:
- `engine/`: removed already; it was a demonstrator/verifier harness, not a real port.
- `TODO.md`: removed already; stale and contradicted the real state.
- `frontend-rss/`: prototype/demo code, explicitly not source-shaped tinygrad port work.
- `docs/ARCHITECTURE.md`: stale plan/status that overstates prototype work.
- `port-rss/PORT_STATUS.md`: stale status document; this audit replaces it.
- `port-rss/mnist_train.rss`, `port-rss/mnist_run.py`: special linear MNIST demo, not the
  real `examples/beautiful_mnist.py` path and not a source-shaped port file.
- Python `__pycache__` artifacts.

Keep:
- `tinygrad-rss/`: integrated target.
- `tinygrad-rss/vendor/tinygrad/runtime/autogen/`: exact generated upstream autogen copy, not
  included in RSS sources.
- `port-rss/`: validated migration material.
- `oracle/`: small MC/C oracle support, useful for backend verification, but not counted as port.
- `port-rss/BRIEFING.md`: useful RSS/MC language rules.

## Recommended Next Order

1. Continue `uop/*` integration: complete remaining `UOp` methods, symbolic/decomposition
   coverage, `spec.py`, and `validate.py` against the upstream source layout.
2. Continue Tensor mixin integration: full dtype/bitcast coverage, remaining random APIs, full
   lazy indexing, higher ops, complete conv/pool options, and exact public method semantics.
3. Integrate schedule/render/codegen into one executable backend path.
4. Add package-level differential tests against real tinygrad for each integrated subsystem.
5. Only then attempt real examples such as `examples/beautiful_mnist.py`.
