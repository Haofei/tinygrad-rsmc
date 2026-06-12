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
  - current validation gate also runs `oracle/roundtrip.py`, clang syntax checks for the generated
    two-output C files, the generated executable smoke, and `git diff --check`.
- every retained standalone `port-rss/*.rss` file runs today:
  - 55/55 RSS files passed with a 25s per-file timeout. The removed MNIST demo also passed,
    but was deleted because it was not a source-shaped tinygrad port.
- source size inventory:
  - tinygrad handwritten Python, excluding `runtime/autogen`: 118 files, about 33k LOC.
  - tinygrad `runtime/autogen`: 88 generated files, about 179k LOC.
  - integrated `tinygrad-rss/src`: 33 RSS files, 28,642 LOC.
  - vendored `tinygrad-rss/vendor/tinygrad/runtime/autogen`: 88 generated Python files,
    exactly copied from upstream tinygrad commit `fa400f9790ab9a684387b02e958658217b33e7c1`.
  - standalone `port-rss`: 55 RSS files, about 12.1k LOC.
- rough source coverage inventory:
  - command: `python3 tools/port_coverage.py --limit 8`
  - result: `tensor.py` 103/106 symbols, `mixin/__init__.py` 82/82 symbols,
    `uop/ops.py` 216/221 symbols; 401/409 total rough symbols covered.
  - this is a batching compass only; symbol presence does not prove exact 1:1 semantics.

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
- RSScript now exposes byte/list conversion and byte-oriented SHA3/SHAKE helpers
  (`Bytes.from_uints`, `Bytes.to_uints`, `Hash.sha3_224_bytes`, `Hash.sha3_256_bytes`, and
  `Hash.shake128_bytes`) across stdlib interfaces, runtime ABI, reg VM, and REIR capability
  classification for tensor hash parity work.
- RSScript now exposes `HttpResponse.bytes` across stdlib interfaces, runtime ABI, reg VM, and
  Rust runtime so URL-backed tensor creation can consume response bodies as bytes instead of
  lossy text.
- RSScript sync `Http.get` now drives the existing reqwest/tokio implementation through the
  native pending executor and preserves raw response bytes alongside text, so current
  `from_url` can execute real byte downloads when network access is available.

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
  `INDEX` with explicit pointer-vs-value metadata, `LOAD`/`STORE`, codegen helper nodes (`CUSTOM`/`CUSTOMI`, `INS`, `GETADDR`,
  `WMMA`/`SHAPED_WMMA`, `VCAT`/`PTRCAT`), and ordering nodes (`WAIT`, dtype-preserving
  `AFTER`, `END`, `BARRIER`), matching the role of upstream
  `tinygrad/uop/ops.py`. REDUCE metadata is
  explicit (`sum`/`prod`/`max` kind plus axes) rather than encoded as sentinel axes.
  `BufferizeArg` device metadata is inspectable for scheduler validation and rewrite tests, and
  value-op effective device projection now follows upstream's first-device-carrying-source behavior
  for ALU/movement/order/load-store/reduce-style nodes.
- `uop/methods.rss`: integrated UOp helper surface over interned node ids: `const_like`, `ufix`,
  masked-index `invalid`/`valid`/`get_idx`/`get_valid`,
  symbolic variable `expr`/`unbind`/`val`/sorted `variables`/`unbind_all`,
  range string/axis helpers, upstream-style `axis` propagation for sharded `PARAM`/`MULTI`/`GETTUPLE`/ALU/reduce/permute,
  canonical device key/count helpers for `PARAM`/`DEVICE`/`STAGE`/`BUFFER`/`COPY`/`ALLREDUCE`/`MSELECT`/`MSTACK`,
  source-aligned `addrspace` key propagation for `PARAM`/`BUFFER`/`DEFINE_LOCAL`/`DEFINE_REG`/`LOAD`,
  pointer/movement propagation, and all-same `STACK`/elementwise/`WMMA` merging,
  param/addrspace display helpers, source-aligned vector-dtype `broadcast`/`STACK`, full-dtype
  `gettuple`, `sink`/`maketuple`/`group`/`vectorize`/`cast`/`bitcast`/`gep`,
  source-shaped `load`/`store`/`wait`/`end`/`after`/`barrier`, `split_uop`, movement `base`,
  `multibase`, scheduler-facing `buf_uop` projection through movement/`AFTER`/`MSELECT`/`MSTACK`,
  source-shaped `multi`/`copy_to_device`/`allreduce`/`mselect`/`mstack`/`detach`/`contiguous_backward`,
  source-aligned `has_buffer_identity`, `max_shape`, `max_numel`, `shard_shape`,
  `max_shard_shape`, `device`, `addrspace`, `new_buffer`, `empty`, `empty_like`, `clone`,
  `shard`, non-realized buffer status accessors over the current RSS runtime model,
  `unique`, `from_buffer`, `_frompy`, `variable`, `bind`, `placeholder`, `placeholder_like`,
  `param`, and `param_like`,
  source-compatible symbolic/introspection aliases for `identity_element`, `simplify`/`ssimplify`,
  `srender`, `range_str`, `multirange_str`, `smax`, `smin`, `resolve`, `vmin`/`vmax`,
  `overflows`, `backward_slice_with_self`, `topovisit`, `argstr`, and `ins`, mapping to the
  existing graph rewrite, render, min/max, range, traversal, and `INS` primitives,
  construction/program-helper wrappers for `sym_infer`, `consumer_map_from_toposort`, `sintify`,
  scalar eval coercions, generic `f`, `index`, `contract`, `range`, `special`, `reduce`/`_rop`,
  register definition, `bounds`, `metadata`, `sgep`, `marg`, `gcd`, `_suop`, ProgramInfo-shaped
  `function_name`/`runtimevars`/`launch_dims`/`vals`/`from_sink`, and integer `safe_exp2`,
  `safe_pow`, `exec_alu`, `bitcast`, and `get_location` compatibility helpers,
  source-compatible structural/property aliases for `key`, `tagstr`, `tuplize`, `ptrdtype`,
  `trace_num`, `rtag`, `_sym_fxn`, `_unshard`, `_mop`, `contiguous_view_offset`,
  `sint_to_uop`, `to_max_shape`, `select_dtype`, `_index_to_concrete_int`, `gate_kernel_sink`,
  and `do_unbind`; these are grounded in the current interned-DAG helpers, but tag metadata is
  still not represented, `key`/`tuplize` are structural strings rather than Python bytes/tuples,
  and contiguous-view offset only proves the conservative zero-offset cases,
  `custom_kernel` forwarding to the existing `CUSTOM` function node constructor,
  source-aligned `contiguous` no-op behavior and `bufferize` as `STAGE`, concrete `as_shape`, source-shaped movement wrappers
  `reshape`/`expand`/`permute`/`flip`/`shrink`/`pad`,
  high-level `call` lowering to `CALL` or `FUNCTION` and `set` lowering to `STORE`/`END`/`AFTER`,
  movement-argument extraction, full-dtype-preserving replace-by-arg/substitute, replace-by-dtype, structural key rendering,
  toposort/backward-slice helpers including upstream `enter_calls=false` body-skipping for `CALL`/`FUNCTION`, shared-parent counting, and source-aligned `ranges`/`ended_ranges` propagation for
  range-carrying and range-ending UOps.
- `uop/upat.rss`: integrated UPat/PatternMatcher over real interned node ids, with named captures,
  capture consistency, op sets, dtype constraints, variadic `allow_any_len`, rules-as-data, and a
  generic bottom-up rewrite/fixpoint driver, plus grouped source-shaped builder helpers for
  unary/ternary ops and `cast`/`bitcast`/`gep`/`load`/`store`/`reduce`/`broadcast`/
  `contiguous`/`after`/`end` patterns, plus source-compatible pattern method wrappers for
  `named`, `or_casted`, `or_after`, `cvar`, `sink`, `index`, `gep`, `load`, `store`, `reduce`,
  `broadcast`, `after`, `end`, and modulo patterns. It also exposes source-compatible rewrite
  driver wrappers for `rewrite`, `pm_rewrite`, `cached_bpm_rewrite`, `walk_rewrite`, and
  `unified_rewrite` over the current `PatternRule` engine, plus UPat compatibility hooks for
  `match`, `_check_dtype`, `_ensure_float`, reverse floordiv construction, callable
  deconstruction/interpret/deferred-compile placeholders, and non-tracing `add_trace_group`,
  `track_rewrites`, and `profile_matches`; Python callable bytecode reconstruction, real deferred
  UPat compilation, and trace/profile collection are still not represented.
- `uop/spec.rss`: first integrated interned-graph verifier for the shared core currently built by
  the package: CONST/SPECIAL/RANGE, `DEFINE_VAR`/`BIND`, `PARAM`, `DEFINE_LOCAL`/`DEFINE_REG`,
  upstream-shaped device/buffer/copy/multi-device graph nodes, ALU dtype rules, same-itemsize `BITCAST`, WHERE/MULACC, buffers, movement
  nodes, STACK/SINK/MSTACK, source-shaped program/call/stage/slice/codegen-facing nodes,
  codegen helper `CUSTOM`/`CUSTOMI`, `INS`, `GETADDR`, `WMMA`/`SHAPED_WMMA`, and `VCAT`/`PTRCAT` nodes,
  `UNROLL`/`CONTRACT` dtype-count/product validation including void `CONTRACT` for store contraction,
  tuple/gettuple, index/load-store, source-shaped `REDUCE`/`STORE` range tails,
  control-flow `IF`/`ENDIF`, order nodes including `END` range validation, symbolic `STACK`
  movement shape args, and recursive source validation.
- `uop/validate.rss`: first integrated source-shaped `tinygrad/uop/validate.py` slice for masked
  index bounds validation over the interval subset already covered by `uop_min_max`: static false
  gates are accepted, active/unknown gates require the index interval to be fully inside
  `[0, size)`, `INDEX` nodes can be validated by deriving flat buffer size from the base shape
  and reading the index/gate sources, and the local `validate_index_with_z3` entry point aliases
  this conservative interval proof. Full Z3-backed boolean/arithmetic solving remains unported.
- `uop/render.rss`: first integrated source-shaped `tinygrad/uop/render.py` slice over interned
  UOp ids, covering the core symbolic renderer (`DEFINE_VAR`, named `SPECIAL`, `PARAM`, `RANGE`, `CONST`,
  `CAST`, `BIND`, unary/binary/ternary ALU render rules, `INDEX`/`STAGE`, and `STACK`),
  inference-specific rendering for div/mod and `BITCAST`, compact UOp line printing, and a first
  `pyrender`-style reconstruction slice for constants, variables, params, casts/bitcasts, binary
  ALU expressions, `WHERE`, and `INDEX`.
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
  `CMPEQ`, plus `THREEFRY(uint64,uint64)` lowering to primitive uint32/uint64 add/xor/shift/cast/or
  nodes for RNG support. The large transcendental, long-integer, dtype, fast-idiv, broader boolean/comparison
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
- `codegen/late/regalloc.rss`: first source-shaped boundary slice for
  `tinygrad/codegen/late/regalloc.py`, covering upstream's pseudo-op set
  (`CONST`, `NOOP`, `AFTER`, `BARRIER`, `GROUP`) and an explicit capability predicate that keeps
  real linear-scan allocation disabled until the RSS IR carries ISA `Register` tags from a target
  renderer.
- `codegen/late/gater.rss`: integrated source-shaped `tinygrad/codegen/late/gater.py`
  slice, covering the `pm_move_gates_from_index` rewrites for 1D and 2D
  `INDEX(WHERE(gate, idx, INVALID))` nodes feeding `LOAD` or `STORE`, including optional
  pointer casts. The port rebuilds the index with the raw idx values, moves the validity
  condition onto the load/store gate, supplies a zero-like load fallback, combines an existing
  store gate with the moved index gate, and folds `WHERE` around gated loads into the load alt
  value for direct and inverted gate forms.
- `codegen/late/expander.rss`: first integrated source-shaped `tinygrad/codegen/late/expander.py`
  slice, covering axis-size choice/index helpers and the compact normalization rewrites for
  `CONTRACT`, empty/double `UNROLL`, and `END` consuming `UNROLL` axes. It also has the common
  same/mixed-axis `do_expand` path for non-WMMA roots, including `UNROLL` source stripping,
  mixed-axis `GEP` swizzles, scalar broadcast, vector `VCAT` repetition, range-arg passthrough,
  and `GEP` arg expansion. It also has the first `pm_pre_expander` slice: `UNROLL`/`UPCAST`
  ranges become `UNROLL` constants, and `REDUCE`/`STORE` nodes carrying `UNROLL` tails are
  contracted before later expansion. WMMA,
  register-pointer special-casing, group-for-reduce, and BufferizeOpts-local
  rewrites remain later expander slices.
- `codegen/late/devectorizer.rss`: first integrated source-shaped
  `tinygrad/codegen/late/devectorizer.py` slice, covering the `devectorize_alu` scalarization
  rule for vector ALU, `CAST`, `BITCAST`, and the local flat-metadata `WMMA` shape: vector operations are rebuilt as scalar lane
  operations over `GEP` sources and wrapped in a `STACK`, plus the upstream
  no-range horizontal `REDUCE` lowering for `ADD`, `MUL`, and `MAX`. It also has the upstream
  `CAST(AFTER(x, deps...)) -> AFTER(CAST(x), deps...)` ordering rewrite. It also has the first `pm_render`
  normalization rules: vector `CONST` nodes become scalar-constant `STACK`s, multi-lane `GEP`
  becomes a `STACK` of lane `GEP`s, `GEP(x, 0)` on scalar `x` unwraps, and one-element `STACK`
  unwraps. The context-free `load_store_folding` rules for `INDEX(STACK(bufs), vec_idx)`,
  `GEP` after `LOAD`, `GEP` on `STORE`, `PTRCAT` after `LOAD`, and `PTRCAT` after `STORE`
  are also integrated, along with the `pm_add_loads` rules that insert `LOAD` for
  non-pointer `INDEX` nodes and clean nested `LOAD(LOAD(ptr))` and `STORE(LOAD(ptr), value)`.
  Folded-index regrouping, load/store splitting,
  buffer/index devectorization, range-backed REDUCE-to-acc lowering, image rewrites, and
  deeper image add-load rewrites remain later devectorizer slices.
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
  tuple-element shape propagation through `TUPLE` and `FUNCTION`, upstream-style global shape for
  `MULTI` by expanding the payload shard axis by device count, boundary-node shape pass-through
  for `CONTIGUOUS`/`CONTIGUOUS_BACKWARD`/`DETACH`/`COPY`/`ALLREDUCE`/`MSELECT`/`MSTACK`,
  vector-shaped scalar/control helper nodes, broadcasting, and View/ShapeTracker core for
  contiguous views, permute, flip, expand, pad, shrink, flat-index expression, and contiguity
  checks.
- `schedule/__init__.rss`: first integrated source-shaped `tinygrad/schedule/__init__.py` slice:
  `_unwrap_src`-style input unwrapping, `_split_after` for `AFTER` kernels/dependencies, dependency
  edge construction from `CALL`/`END` inputs and nested `AFTER` nodes, topological emission of a
  `LINEAR` node, and source-shaped rebuilding of emitted `CALL`s with non-`BIND` inputs converted
  through `buf_uop`. It also has the first `CALL(LINEAR, args...)` resolver, replacing scheduled
  `PARAM` nodes with call arguments and flattening nested `LINEAR` nodes, plus the first
  `create_schedule_with_vars`-style variable binding extraction primitive: collect variables used
  by emitted linear bodies, keep only matching `BIND` values from the sink, ignore unused binds,
  and reject conflicting duplicate values. The first supported `create_linear_with_vars` wrapper now
  connects direct `LINEAR` roots or `CALL(LINEAR, args...)` roots to linear-call resolution, held
  buffer collection, variable extraction, and the current memory planner. Full upstream
  `pm_schedule` graph rewriting, schedule caching, rangeify graph generation, complete linear-call
  resolution, exact memory planning integration, and JIT capture plumbing remain unported.
- `schedule/memory.rss`: first integrated source-shaped `tinygrad/schedule/memory.py` slice:
  buffer collection through `BUFFER`/`MSELECT`/`MSTACK`, held-buffer and disk/tinyfs rejection,
  first/last lifetime tracking, copy-vs-compute lane separation, per-device/per-lane int8 arenas,
  and buffer-to-`SLICE` rewrite for planned `LINEAR` calls. Exact TLSF lifetime reuse remains
  unported; the current integrated planner uses monotonic lane offsets.
- `schedule/indexing.rss`: first integrated source-shaped `tinygrad/schedule/indexing.py` slice:
  `ALWAYS_CONTIGUOUS` classification, realize-map generation for `COPY`/`CONTIGUOUS`/`STORE`,
  non-contiguous source realization for `COPY`/`MSELECT`/`MSTACK`, direct `COPY`/`SLICE` store-source
  cleanup, simple store self-dependency hazard detection, range-context allocation with size-1
  folding, first `apply_movement_op` mappings for `SHRINK`, `PERMUTE`, `FLIP`, `EXPAND`,
  validity-guarded `PAD`, and row-major `RESHAPE` div/mod coordinate splitting with identity
  fast-path and per-coordinate symbolic rewriting, plus the first
  `run_rangeify` analysis half for realized output ranges, single-consumer inheritance,
  multi-consumer same-index validity merging with per-axis realization fallback,
  EXPAND-origin ending-range propagation and default-PCONTIG elementwise/reduce realization,
  movement input-range swizzling, and `REDUCE` axis range injection, plus the first graph rewrite
  half for source `INDEX` insertion, realized `STAGE` insertion, movement removal, and `REDUCE`
  range-source conversion. Graph-backed `PAD`/`SHRINK` movement args are recovered from shape
  sources, `PAD` rewrites now emit the upstream local `WHERE(valid, value, 0)` form, and
  realized child staging now follows upstream global-vs-local, removable, and full-global device
  option rules.
- `schedule/allreduce.rss`: first integrated source-shaped `tinygrad/schedule/allreduce.py` slice:
  naive allreduce graph construction for supported multi-device `MULTI`/`MSTACK` inputs by
  normalizing the input through `CONTIGUOUS`, selecting/copying each shard to the target device,
  and reducing copied shards with `ADD`/`MUL`/`MAX`. Ring/all2all chunking and allreduce function
  wrapping remain unported.
- `schedule/multi.rss`: first integrated source-shaped `tinygrad/schedule/multi.py` slice:
  early `PARAM -> MULTI` lowering for axis-tagged tuple-device params, COPY/MSELECT/MSTACK rewrite helpers for broadcasting a single-device COPY to a tuple
  device as `MSTACK(copy...)`, copying a multi-device value to one device by copying each shard and
  concatenating along the sharded axis, copying a multi-device value to a tuple device through
  symbolic unshard plus `ALLREDUCE(ADD)`, eliminating `MSELECT(MSTACK)`, moving `MSELECT` before movement ops, passthrough
  rebuilds for boundary ops with a leading `MULTI` child, source stripping for non-value-producing
  roots such as `STORE` and void `CALL`, multi-aware `AFTER(MULTI, STORE(MULTI, MULTI))` ordering, and the first movement rewrite family for
  `RESHAPE`/`EXPAND`/`PAD`/`SHRINK`/`PERMUTE`/`FLIP`, nonzero shard-partition `SHRINK(MULTI)` lowering through
  `MSELECT` plus tuple-device `COPY`, plus early `SHRINK(MSTACK)` splitting and tuple routing for
  `GETTUPLE(TUPLE)` and `GETTUPLE(MULTI(TUPLE|FUNCTION))`, FUNCTION-body multi stripping/rewrapping, and source-shaped `REDUCE(MULTI)`
  handling for piecewise and shard-axis allreduce reductions, explicit tuple-device
  `ALLREDUCE(MULTI)` passthrough, plus upstream-shaped unary/binary `ALU(MULTI, ...)` payload
  rewrites including scalar-broadcast operands, non-scalar unsharded operands through symbolic
  `_device_num` shard bounds, and axis-mismatch unshard/reshard through symbolic `PAD`,
  tuple-device `ALLREDUCE(ADD)`, and symbolic `SHRINK`.
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
  staging for unsupported shapes. A first scheduler-to-realize bridge now runs supported scheduled
  sinks by calling `schedule_create_linear_with_vars` and then `run_linear`; resolved concrete
  buffer arguments fall back to their call-argument position when no `PARAM` slot remapping is
  needed. Extracted scheduled `var_vals` are threaded into PROGRAM staging, where ProgramArg
  values now record total, known-bound, and runtime-var counts. Local-size optimization,
  dynamic symbolic launch dimension evaluation, general compiled numeric kernel invocation,
  validation execution, graph execution, full multi-buffer/device remapping, and dynamic-library
  runtime plumbing remain unported.
- `gradient.rss`: first integrated reverse-mode autodiff slice over interned UOp ids, covering
  target-specific symbolic gradients for `CAST`, `ADD`, `SUB`, `MUL`, `FDIV`, unary
  `NEG`/`RECIPROCAL`/`SQRT`/`EXP2`/`LOG2`/`SIN`/`TRUNC`, binary `POW`, `MAX` with upstream
  tie-splitting, mask-selection `WHERE`, zero gradients for integrated comparison predicates, and
  movement/reduce VJPs for `RESHAPE`, `EXPAND`, `PERMUTE`, `FLIP`, `PAD`, `SHRINK`, and `REDUCE`,
  plus grouped boundary VJPs for `CONTIGUOUS`, `CONTIGUOUS_BACKWARD`, `COPY`, `DETACH`, and `BITCAST`, then
  simplifying the resulting gradient graph.
- `uop/vminmax.rss`, `uop/alu.rss`, `uop/symbolic.rss`, `runtime/ops_python.rss`: small
  interpreter/simplifier pieces. The evaluator now passes through boundary data movement nodes
  `CONTIGUOUS`, `CONTIGUOUS_BACKWARD`, `DETACH`, and `COPY`. The symbolic layer now includes integer vmin/vmax, rewrite
  identities/folding, a grouped `symbolic_simple` slice for idempotent ops, boolean constants,
  same-self comparisons, `x//-1`, `(x^y)^y`, boolean not/cast/trunc folding,
  bool-ALU-to-logic folding, boolean contradiction folding, additive coefficient combining,
  constant-times-sum distribution, exact min/max constantization, bound-proven `MAX` folding,
  associative constant-chain folding, comparison threshold normalization, chained floor-div
  folding, constant-last reordering, range/end division folding, cast-chain folding,
  `AFTER` dependency cleanup, `GROUP`/`SINK` flattening, vectorized binary ALU lane splitting,
  `CAST(WHERE(...))` pushdown, reciprocal and weakint distribution algebra,
  vector `STACK(CONST...)` folding, inverted-condition `WHERE` folding, simple `WHERE` folding,
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
  `kaiming_normal`), upstream-shaped scalar/wrapper helpers (`_uop`, `_wrap_uop`,
  `const`, `const_like`, `unique_const`, `_broadcasted`, and `_binop`),
  upstream-shaped higher-op wrappers (`__rmatmul__`, single-axis and trailing multi-axis
  `interpolate`, `_apply_ceil_mode`,
  `avg_pool2d`, `max_pool2d`, `max_unpool2d`, `conv2d`, `conv_transpose2d`,
  `batchnorm`, `_do_reduction`, `binary_crossentropy`, `binary_crossentropy_logits`,
  `sparse_categorical_crossentropy`, `cross_entropy`, and `nll_loss`) over the existing
  explicit RSS tensor kernels,
  dtype conversion through `CAST` for core float/int/bool cases, same-itemsize
  `BITCAST` graph construction plus upstream-style shape-changing bitcast composition through
  unsigned integer packing/unpacking, broadcast elementwise arithmetic/comparison ops, unary
  `EXP2`/`LOG2`/`SIN`/`SQRT`/`RECIPROCAL`/`NEG`/
  `TRUNC`, binary `POW`, composed `rsqrt`, `log10`, `minimum`, `square`, `clip`, `sign`, `abs`,
  `ceil`, `floor`, `sigmoid`, `exp`, `log`, `cos`, `tan`, `tanh`, `leaky_relu`, `quick_gelu`, default
  tanh-approx `gelu`, `swish`/`silu`, `hardswish`, `hardsigmoid`, `softplus`, `mish`,
  `logsigmoid`, `elu`, `celu`, `selu`, `softsign`, `lerp`, `hardtanh`, `relu6`,
  explicit-seed `dropout`, no-mask `scaled_dot_product_attention`, and polynomial
  `newton_schulz`,
  bitwise/logical `AND`/`OR`/`XOR`, integer shifts, floor/truncating div and modulo variants,
  `WHERE` mask selection, movement helpers (`reshape` with single `-1`
  inference, `view`, `permute`, `flip`, `pad`, `pad_to`, `expand`, `shrink` through explicit
  per-axis `slice`, `shrink_to`, `split`, `chunk`, `meshgrid` for 1D/scalar tensors, and
  `diag` for 1D tensors, `diagonal`, `triu`, `tril`, `_pad_circular`,
  `_pad_reflect_replicate` for reflect/replicate-style padding over movement primitives,
  `repeat_interleave`, `repeat`, `roll`, `unfold`, `cat`,
  `stack`, ADD-backed `cumsum`, MUL-backed `cumprod`, MAX-backed `cummax` values/indices,
  negated-MAX-backed `cummin` values/indices, and upstream-shaped stable `logcumsumexp`),
  composed movement helpers (`unsqueeze`, `squeeze(dim)`,
  `squeeze()`, `transpose`, `flatten`, and `unflatten`), reduce helpers (`sum(axis)`, `sum()`,
  `prod(axis)`, `prod()`, `max(axis)`, `max()`, `min(axis)`, `min()`, bool `any`/`all`,
  `mean(axis)`, `mean()`,
  axis/all `var` and `std`, tuple-axis `mean`/`var`, single-axis `normalize`, single-axis and
  tuple-axis `layernorm`, channel-axis `batchnorm` with no-affine and affine forms, plus
  keepdim-backed `logsumexp(axis)`, `softmax(axis)`, `log_softmax(axis)`, first-index
  `argmax`/`argmin` for all/axis reductions, and graph-composed stable `sort`, `argsort`,
  and sorted `topk` for fixed-shape tensors), graph-composed `isnan`/`isinf`/`isfinite`
  predicates (`isnan` as `x != x`, `isinf` as positive/negative infinity equality, and
  `isfinite` as the negation of NaN/Inf) plus
  `isclose`/`allclose` with `equal_nan` handling, loss helpers (`binary_crossentropy`,
  `binary_crossentropy_logits`, `nll_loss`, and sparse-target `cross_entropy`), `_pool`-style
  NCHW `avg_pool2d` with count-include/count-exclude padding and `ceil_mode`, `max_pool2d` with
  low constant padding, dilation, `ceil_mode`, padded/dilated/ceil `return_indices`, default and
  explicit-output `max_unpool2d`, `conv2d` with optional bias, zero/signed padding, dilation, and grouped convolution,
  and NCHW `conv_transpose2d` with stride, padding, dilation, output padding, bias, and groups,
  graph-backed `_one_hot_along_dim`, `one_hot`, generic `gather(dim, index)`, plain `scatter`
  with duplicate last-wins masked merge, `scatter_reduce` for sum/prod/mean/amax/amin,
  scalar `scatter(..., reduce="add"/"multiply")`, upstream-shaped wrappers for `_pre_scatter`,
  `_masked_merge`, `scatter_reduce`, `_tri`, `_ufix_keep_dtype`, constrained materialized
  `__getitem__`/`_getitem`, implicit-scalar and explicit-incoming-gradient `gradient`/`backward`
  wrappers over the integrated `grad_wrt` slice, `_apply_uop` dispatch for movement/reduce/unary/binary/where graph
  construction, Tensor `_rop` reduce forwarding, source-shaped `sequential` over graph function
  UOps, plus boundary wrappers for `alu`, scalar `ufix`, `contiguous`, and `to` over existing UOp
  graph primitives. The RSS `backward` wrapper returns gradients for explicit target UOps and has
  explicit-state `TensorState` helpers to set, clear, discover, and accumulate `.grad`; it does
  not yet mutate Python Tensor objects in place or discover targets through a weakref registry,
  constrained object-state helpers for `__hash__`, `get`/`set`-style state access,
  whole-tensor and indexed `__setitem__` graph assignment over the existing `assign`/`getitem`
  helpers, `__delitem__` rejection as a false/unsupported result, and a metadata-state wrapper
  record, plus materialized 2D `qr` and thin `svd` helpers over the current evaluator-backed
  tensor data path. `qr` uses Gram-Schmidt and returns full `Q` and `R`; `svd` uses a Jacobi
  eigensolve of `A^T A`, returns thin `U`, `S`, and `Vt`, and accepts but does not yet implement
  `full_matrices`. These are graph/state compatibility helpers, not full Python object identity,
  exception, weakref, or context-frame metadata semantics,
  a first graph-level `TensorState` lifecycle spine with `uop`/param/grad flags, state
  `replace`/`as_param` helpers, graph-backed `shape`/`dtype`/`device`/`numel`/`nbytes`,
  `grad`/`has_grad`/`clear_grad`/`set_grad`/accumulate helpers and batch state `backward`
  wrappers that update explicit RSS `TensorState` lists,
  `empty`/`empty_like` over UOp buffer construction, no-op-aware same-device `to`, single-tensor
  `realize` through the integrated scheduler/runtime path, and narrow evaluator-backed
  `data`/`item`/`tolist` helpers for supported CPU/value graphs,
  graph-level `call`, `callify`, `linear_with_vars`, and `schedule_linear` wrappers over the
  existing `FUNCTION`/`CALL` and scheduler primitives, simple `assign` as STORE+AFTER with
  broadcast-to-target-shape, `clone` preserving param/grad state, `shard`/`shard_`/`shard_like`
  over the integrated multi-device UOp helpers, and in-place binary/matmul wrappers composed as
  `op` plus `assign`,
  an explicit RSS-side `TensorRngState`/counter wrapper for `manual_seed` and `_next_counter`,
  multi-device `rand_like` construction through `MSTACK`/`MULTI`, and materialized 1D/2D
  `multinomial` sampling for supported CPU/value tensors,
  Tensor boundary helpers for invalid-sentinel creation (`invalids` over `uop_invalid`), shape/dtype/device
  `repr`, scalar-shape `len` rejection via sentinel return, bool rejection flag, contiguous-buffer graph
  boundary, and list-backed `numpy`/host-data materialization over the current evaluator,
  materialized `keccak` for row-wise `sha3_224`, `sha3_256`, and `shake_128` byte tensors plus
  a materialized `_hash_1mb` SHAKE-128 chunk/reduce helper over the current evaluator byte path,
  materialized byte/file boundary helpers for current local execution (`fs_load`/`fs_store`
  through `File.read_bytes`/`File.write_bytes`, `from_blob` as an external-pointer-shaped empty
  buffer placeholder, `from_url` through sync `Http.get`/`HttpResponse.bytes` with gzip byte
  decompression through `Gzip.decompress_bytes`,
  and `decode_hevc_frame` as an `encdec` `CUSTOM_FUNCTION` call dependency graph wrapper),
  explicit-size graph-shaped `masked_select` and constrained 1D `nonzero` over scatter/gather
  compaction,
  plus legacy materialized gather indexing (`gather_axis0`), upstream-style `dot`/`matmul`/`linear`
  composition for vector, matrix, and batched cases, upstream-style `relu` as
  `(x > 0).where(x, 0)`, and legacy `linear_forward`.

Current integrated demo:
- validates dtype, helpers, view/shape, UOp helpers, UOp spec checks, UPat matching, and
  rules-as-data graph rewrite.
- validates UOp method helpers for `STACK` splitting/gettuple, source-shaped
  `sink`/`maketuple`/`group`/`vectorize`/`cast`/`bitcast`/`gep`,
  `load`/`store`/`wait`/`end`/`after`/`barrier`, concrete shape extraction,
  movement base recovery, and movement argument extraction.
- validates grouped UPat builder helpers for unary/ternary ops and source-shaped
  `cast`/`bitcast`/`gep`/`load`/`store`/`reduce`/`broadcast`/`contiguous`/`after`/`end`
  patterns.
- validates source-shaped movement method helpers for `reshape`/`expand`/`permute`/`flip`/
  `shrink`/`pad`, scalar empty-argument no-op behavior, full-dtype preservation through
  movement constructors, and upstream-shaped weakint vector `shape_to_shape_arg`.
- validates high-level UOp sugar for upstream-shaped `call` dispatch to opaque `CALL` vs
  value-producing `FUNCTION`, and `set` lowering through `STORE`/`END`/`AFTER`.
- validates grouped gradient boundary rules: `CONTIGUOUS`/`COPY` pass gradients through,
  `CONTIGUOUS_BACKWARD` wraps the adjoint in `CONTIGUOUS`, and `DETACH`/`BITCAST` stop gradients.
- validates the first source-shaped scheduler dependency slice: `AFTER` splitting, movement unwrap
  back to `AFTER`, dependent `CALL` ordering into a `LINEAR` node, and rebuilt `CALL` buffer args.
- validates the first post-schedule linear-call resolver: `CALL(LINEAR, arg)` replaces slot-0
  `PARAM` with the call argument and flattens nested `LINEAR` nodes.
- validates the first scheduler variable binding extraction primitive: only variables used by
  emitted linear bodies are collected from `BIND` sink children, unused binds are ignored, and
  conflicting duplicate values reject the extraction.
- validates the first supported `create_linear_with_vars` wrapper over a direct `LINEAR` root:
  used binds are returned, unused binds are ignored, and a concrete buffer argument is rewritten
  through the current monotonic-offset memory planner. It also validates the `CALL(LINEAR, arg)`
  root path: slot-0 `PARAM` is resolved to the outer call argument and the held concrete buffer is
  not rewritten by memory planning.
- validates the first scheduler-to-realize bridge: supported scheduled `CALL(LINEAR, ...)` roots
  resolve inner `PARAM` args, run through `realize_run_linear`, and execute COPY, SLICE/view, and
  hosted PROGRAM calls with expected `TGBuffer` bytes. It also validates PROGRAM `var_vals`
  accounting for direct host execution and scheduled PROGRAM execution with a `BIND`.
- validates the first source-shaped scheduler memory planner slice: compute/copy lane separation,
  int8 arena `SLICE` rewrites that pass the integrated spec verifier, and held-buffer exclusion.
- validates the first source-shaped scheduler indexing slice: `ALWAYS_CONTIGUOUS` classification,
  unconditional realize ops, non-contiguous copy-source realization, direct copy-store cleanup, and
  self-store source realization.
- validates scheduler indexing range/movement helpers: size-1 range folding, sequential range ids,
  `SHRINK`/`PERMUTE`/`FLIP`/`EXPAND` index mapping, validity-guarded `PAD` index mapping,
  and row-major `RESHAPE` div/mod index mapping plus identity simplification.
- validates scheduler indexing rangeify analysis records for realized roots, inherited movement
  chains, `PERMUTE` input-range swizzling, same-index multi-consumer range merging without
  unnecessary staging, EXPAND-triggered ending-range realization, and `REDUCE` `AXIS_REDUCE`
  injection.
- validates scheduler indexing rangeify graph rewrite for movement removal through `INDEX` and
  `REDUCE` metadata-to-range-source conversion, plus PAD-to-`WHERE` locality wrapping and
  device-carrying full-global `STAGE` option propagation.
- validates scheduler indexing rangeify `STAGE` option selection for full global staging,
  partial local staging, and upstream removable rules for always-contiguous vs non-contiguous children.
- validates vector-dtype `broadcast`, full-dtype `GETTUPLE`, full-dtype-preserving
  `replace_arg`/substitution, and full-dtype-aware `CAST`, including vector-to-scalar
  constructor casts that must not be skipped by scalar dtype-name comparison.
- validates source-aligned masked-index helpers: `valid(idx, cond)` builds a verifier-accepted
  `WHERE`, `get_idx` unwraps the payload, `get_valid` unwraps the mask, and vector-count
  `invalid` preserves weak-index dtype count.
- validates the integrated interval validator for raw index/gate triples and source-shaped
  `INDEX` nodes: in-bounds indexes pass, out-of-bounds indexes fail, and statically false gates
  skip the out-of-bounds proof.
- validates UOp symbolic variables and binding: expression extraction, declared vmin/vmax,
  in-range `BIND`, `unbind`, `val`, sorted `variables`, `unbind_all` rewrite with collected
  bindings, and out-of-range bind rejection.
- validates the integrated source-shaped UOp renderer for precedence-aware symbolic expressions,
  `max`, inference `floordiv`, inference `BITCAST`, compact repeated `STACK`, UOp line output,
  and first `pyrender` reconstruction for an ALU expression and bitcast.
- validates UOp `ParamArg` metadata and definitions: `PARAM` slot/name/addrspace, bounded param
  vmin/vmax, `DEFINE_LOCAL`/`DEFINE_REG` addrspace, and rejection of invalid param bounds or
  negative local slots.
- validates upstream-shaped UOp device/buffer graph metadata: `DEVICE` tuple key/count,
  `UNIQUE`-backed `BUFFER` shape/device count, `COPY`, `MULTI`, `MSELECT`, `MSTACK`, `ALLREDUCE`,
  `CONTIGUOUS`, `DETACH`, `CUSTOM_FUNCTION`, and rejection of empty devices, out-of-range
  `MSELECT`, and invalid `ALLREDUCE` ops.
- validates source-shaped multi-device method helpers for `copy_to_device`, `multi`,
  `mselect`, `mstack`, `allreduce`, `detach`, and `contiguous_backward`, plus full-dtype
  preservation through `COPY`, `CONTIGUOUS`, and `DETACH`.
- validates the first source-shaped scheduler allreduce slice: naive `ALLREDUCE` over both
  `MULTI` and `MSTACK` inputs emits verifier-accepted `CONTIGUOUS -> MSELECT -> COPY` shard
  graphs reduced with `ADD`.
- validates the first source-shaped scheduler multi rewrite slice: tuple-device COPY broadcast,
  axis-tagged tuple-device `PARAM -> MULTI`, multi-device COPY-to-one concatenation, multi-device
  COPY-to-tuple unshard/allreduce, `MSELECT(MSTACK)` elimination, `MSELECT` movement pushdown,
  `SHRINK(MSTACK)` splitting, nonzero shard-partition `SHRINK(MULTI)` selection/copy lowering,
  `CAST`/`CONTIGUOUS`/`AFTER` passthrough, `STORE` and void `CALL` source stripping,
  `AFTER(MULTI, STORE(MULTI, MULTI))` ordering, and
  `RESHAPE`/`EXPAND`/`PAD`/`SHRINK`/`PERMUTE`/`FLIP` movement rewrites, tuple and function
  `GETTUPLE` routing through `MULTI`, FUNCTION-body multi stripping/rewrapping, and piecewise vs. shard-axis `REDUCE(MULTI)` rewrites,
  tuple-device `ALLREDUCE(MULTI)` passthrough, plus same-axis unary/binary `ALU(MULTI, ...)`
  rewrites including scalar-broadcast operands, non-scalar unsharded operands in either source
  order, and axis-mismatch resharding.
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
- validates a grouped upstream symbolic bounds/constant-chain batch: exact `DEFINE_VAR`,
  `SPECIAL(size=1)`, and `RANGE(size=1)` constantization, `MAX` folding by disjoint
  ranges, `(x+2)+3 -> x+5`, `(x*2)*3 -> x*6`, `2+x<7 -> x<5`,
  `(x//2)//3 -> x//6`, `3*x<10 -> x<4`, `x//2<5 -> x<10`, and
  `(x+2)+y -> (x+y)+2`.
- validates a grouped upstream symbolic cleanup batch: `RANGE%end -> RANGE`,
  `RANGE//end -> 0`, lossless cast-chain collapse, range-proven integer cast-chain collapse,
  `AFTER` dependency flattening, and vector `STACK(CONST...) -> CONST` folding while preserving
  full vector dtype through `graph_rewrite` and `pattern_graph_rewrite`.
- validates a grouped upstream symbolic phase-three batch: singleton `GROUP` collapse,
  `SINK(GROUP(...))` flattening, vectorized binary ALU lane splitting,
  `CAST(WHERE(...)) -> WHERE(CAST(...), CAST(...))`, `x/(1+x) -> 1-(1/(1+x))`,
  `1/(x*x) -> (1/x)*(1/x)`, and weakint `(x+y)*c -> x*c+y*c`.
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
  signed comparison canonicalizations, tight-range equality, logical-not-of-`CMPNE` lowering, and
  structural `THREEFRY` lowering to a primitive two-source `OR` root.
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
- validates the `pm_move_gates_from_index` slice: masked 1D and 2D `INDEX` nodes feeding
  `LOAD` and `STORE` are rewritten to plain indexes plus gates, optional casted pointers are
  preserved, an existing store gate is combined with the moved validity gate, and `WHERE`
  around direct/inverted gated loads is folded into the load alt value.
- validates the first expander slice: scalar `CONTRACT` lowers to `STACK`, `CONTRACT` of
  `UNROLL` lowers to indexed `GEP`, empty `UNROLL` unwraps, double `UNROLL` merges axes,
  `END` consumes `UNROLL` axes through `CONTRACT`, same-axis elementwise `UNROLL` sources are
  expanded, mixed-axis `UNROLL` sources are swizzled through `GEP`, scalar non-unroll sources
  are broadcast, and `GEP` args expand across lanes. It also validates the first
  `pm_pre_expander` rules for unrolled ranges plus REDUCE/STORE unroll contraction.
- validates the first devectorizer slice: vector `ADD`, vector `CAST`, and vector `BITCAST`
  scalarize into `STACK` nodes of scalar lane operations and pass the integrated UOp spec
  verifier. It also validates the first `pm_render` normalizations for vector `CONST`,
  multi-lane `GEP`, scalar `GEP(0)`, and single-source `STACK`, plus the first context-free
  load/store folding rules for stack-backed `INDEX`, `GEP`, and `PTRCAT`.
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
- validates Tensor `isnan`, `isinf`, and `isfinite` values for NaN/+Inf/-Inf/finite inputs,
  including positive-only and negative-only `isinf` detection, and validates Tensor `isclose`
  and `allclose` for loose/tight finite tolerances and NaN/Inf edge cases including
  `equal_nan=True`.
- validates Tensor loss helpers against real tinygrad: binary cross entropy mean/sum, BCE logits
  mean with and without `pos_weight`, NLL loss none/mean, and sparse-target cross entropy none/mean.
- validates Tensor statistics against real tinygrad: `var(axis=1, correction=0/1)`,
  `std(axis=1, correction=0/1)`, `var(correction=0)`, `std(correction=0)`,
  `var_mean(axis=1, correction=0)`, `var_mean(axis=(1,2), correction=0)`,
  `std_mean(axis=1, correction=0)`, and `std_mean(correction=0)`.
- validates Tensor `nbytes` against real tinygrad for float32 and int32 2x3 tensors.
- validates Tensor `dropout` against real tinygrad for explicit-seed training `p=0.25`,
  eval/no-op behavior, and training `p=1` zeroing.
- validates Tensor `scaled_dot_product_attention` against real tinygrad for no-mask/no-dropout,
  causal/no-dropout, boolean-mask, additive-mask, and grouped-query-attention batched cases.
- validates Tensor `newton_schulz(steps=2, params=(2,-1.5,0.5))` against real tinygrad on a
  2x2 float32 matrix.
- validates upstream-shaped Tensor wrapper aliases for `_uop`/`_wrap_uop`, scalar `const`,
  `unique_const`, reverse `_binop` with `const_like`, `_split_cumalu` for ADD/MUL, and pair
  returns for `cummax`/`cummin` on the active tensor smoke.
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
- validates source-compatible Tensor fallback entry points for `_fromnp` over flattened numeric
  data, `image_dot` through `tensor_dot`, and `image_conv2d`/`_conv2d_winograd` through the
  generic direct NCHW convolution path. This is API-surface compatibility for the current backend,
  not upstream packed image kernels or Winograd transform parity.
- validates an explicit-state `_apply_map_to_tensors` equivalent: callers pass `TensorState`
  objects plus old/new UOp ids, affected graphs are detected with `topovisit`, and states are
  returned with recursive graph substitution applied. This does not yet replicate Python weakref
  discovery of all live tensors or distinct `walk=True` substitution semantics.
- validates Winograd matrix helpers: `_get_winograd_matcols` builds dim-major matrix-column
  tensors, and `_apply_winograd_matrix` applies a row-major matrix over the first one or two axes
  using the evaluator-backed data path. This covers the real matrix transform primitive for current
  Tensor data, but not the full upstream lazy expression or final lazy Winograd convolution path.
- validates Tensor `cumsum` against real tinygrad for axis-0 and axis-1 cases, backed by the
  upstream `_cumalu` ADD composition using zero-padding, pooling, and sum.
- validates Tensor `cumprod` against real tinygrad for axis-0 and axis-1 cases, backed by the
  upstream `_cumalu` MUL composition using one-padding, pooling, and product reduction.
- validates Tensor `cummax` against real tinygrad for 1D axis-0 and 2D axis-1 cases, including
  returned values and indices, backed by the upstream `_cumalu` MAX composition plus match/index
  reconstruction.
- validates Tensor `cummin` for the same 1D axis-0 and 2D axis-1 cases, including returned
  values and indices, backed by negating through the `cummax` values/index reconstruction.
- validates Tensor `logcumsumexp` for axis-0 and axis-1 cases against real tinygrad, backed by
  the current upstream stable cumulative-max, triangular-mask, exp/sum/log composition.
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
  `round`, `sinh`, `cosh`, `atanh`, `asin`, `acos`, `atan`, `asinh`, `acosh`, `erf`,
  `leaky_relu`, `quick_gelu`, default tanh-approx `gelu`, `swish`/`silu`, `hardswish`,
  `hardsigmoid`, `softplus`, `mish`, `logsigmoid`, `elu`, `celu`, `selu`, `softsign`, `lerp`,
  `hardtanh`, `relu6`, and upstream-style `relu`, including positive, negative, and zero values.
- validates bool and int bitwise/logical Tensor ops against real tinygrad: `AND`, `OR`, and `XOR`.
- validates integer shift and division/modulo Tensor ops against real tinygrad: `SHL`, `SHR`,
  floor division/modulo, and truncating cstyle division/modulo.
- validates Tensor interpolation against real tinygrad: one-axis `linear`, `linear` with
  `align_corners=True`, `nearest`, and `nearest-exact` interpolation over `[1,4]`, plus trailing
  2D interpolation from `(2,3)` to `(3,2)` for `linear`, aligned `linear`, `nearest`, and
  `nearest-exact`.
- validates Tensor gradients through broadcast and sum: gradient wrt a `[2,3]` input is all ones,
  and gradient wrt a broadcasted `[1,3]` input accumulates to `[2,2,2]`.
- validates Tensor gradients through graph-backed `SHRINK` slicing against real tinygrad:
  `sum(x[0:2,1:3])` routes gradients back to the source as `[0,1,1,0,1,1]` using a `PAD` VJP.
- validates explicit incoming-gradient routing for that slice by seeding the scalar backward pass
  with `3`, and validates explicit RSS `TensorState` backward helpers that set `.grad`, accumulate
  repeated backward calls, and use supplied gradient state.
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
- validates Tensor gradients through boundary UOps with shape-preserving reductions:
  `CONTIGUOUS`, `CONTIGUOUS_BACKWARD`, and `COPY` route `[1,1,1,1]` back to the source, while
  `DETACH` and `BITCAST` route `[0,0,0,0]`.
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
  explicit-seed random, initializer helpers, and a first graph-level lifecycle/state spine are
  partially integrated now, but exact upstream class-global RNG mutation/counter identity,
  Python exception behavior for `__bool__`/`__len__`/`__delitem__`, true ndarray/buffer object parity,
  full Python `Tensor.__init__`/object identity behavior,
  weakref/all-tensor registry updates, Python-object `backward` grad mutation, full view-assign substitution and realized-buffer mutation safety checks, exact global
  seed/counter APIs, broader distribution helpers,
  full lazy/batched Householder QR and full-matrices/upstream Jacobi SVD parity,
  exact lazy hash graph semantics and strict `_hash_1mb` size enforcement,
  exact tinyfs chunk-tree storage semantics for `fs_load`/`fs_store`, real external-pointer buffer
  ownership/lifetime for `from_blob`, exact cache/progress behavior for `from_url`, and
  actual HEVC decode runtime execution behind the `encdec` custom function,
  real packed image convolution kernels and lazy Winograd convolution integration rather than the
  current direct-conv fallback,
  full symbolic/lazy indexing semantics, dynamic-size `masked_select`/`nonzero` paths that depend
  on runtime `.item()` shape discovery,
  broader composed math coverage,
  full conv/pool semantics including more negative/asymmetric edge cases and broader dimensionality,
  Python-facing method breadth,
  and exact tinygrad semantics are still incomplete.
- `function.py` and `gradient.py`: autograd is partially integrated for scalar symbolic UOp
  graphs, simple Tensor movement/reduce graphs including slice/pad, mask selection, max, and
  selected unary/math VJPs. `tensor_gradient` now wraps the supported implicit scalar-gradient
  and explicit incoming-gradient paths for a list of target UOps, `backward` forwards to those
  explicit-target paths, and RSS `TensorState` helpers model `.grad` discovery/set/clear/accumulation
  for explicit state lists, but full Function-style APIs, complete VJP coverage, Python weakref-driven
  Tensor discovery and in-place Python object mutation, and exact tinygrad gradient semantics are still missing.
- `uop/ops.py`, `uop/upat.py`, `uop/symbolic.py`, `uop/divandmod.py`, `uop/decompositions.py`, `uop/render.py`, `uop/spec.py`, `uop/validate.py`: full
  source-aligned implementation. UPat/rewrite and a broader method-helper/alias slice are integrated now, but
  the full upstream method surface, symbolic/decomposition/divmod coverage, exact symbolic shape
  representation, real upstream `Buffer`/`MultiBuffer` realization semantics, complete render/pyrender
  behavior, full spec validation, full Z3-backed bounds validation, and exact upstream semantics are still incomplete.
- `schedule/*`: source-shaped scheduler remains partial. The first dependency-ordering slice in
  `schedule/__init__.rss` and first monotonic-offset memory planner slice in `schedule/memory.rss`
  are integrated, and `schedule/indexing.rss` has realize-map generation, first range/movement
  helpers including PAD and RESHAPE, first RESHAPE identity/per-coordinate symbolic
  simplification, first rangeify analysis records including same-index multi-consumer merging,
  first EXPAND-origin ending-range realization,
  first device-aware full-global staging options, and first rangeify graph
  rewrite to `STAGE`/`INDEX`. Exact TLSF reuse, full upstream sink-level RESHAPE simplification,
  multi-kernel behavior, schedule caching, linear-call resolution, variable binding extraction,
  and JIT capture plumbing remain.
- `codegen/*`: full lowerer and late passes aligned to tinygrad. The first linearizer priority
  toposort, CFG/control-flow insertion, split-end cleanup, gated-store line cleanup,
  gate-from-index movement, first expander CONTRACT/UNROLL/END normalization, same/mixed-axis expansion, and `pm_pre_expander` REDUCE/STORE/range unroll handling,
  `PROGRAM` -> `PROGRAM+LINEAR` wrapper, integer upper-bound `do_estimates` metadata,
  first `ProgramInfo.from_sink` metadata derivation including integer `SPECIAL` launch dimensions, and
  `PROGRAM+LINEAR` -> `PROGRAM+LINEAR+SOURCE` CStyle wrapper are integrated, but pre-existing
  IF rejection, remaining late expansion/devectorization, range simplification, GPU dims, ISA/regalloc,
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

1. Speed up the port by batching upstream slices: use `tools/port_coverage.py` to generate a
   function/symbol inventory diff for one upstream module or feature cluster, port the dependency
   closure together, then use compiler failures and real tinygrad oracle checks to find
   language/runtime gaps once per batch.
2. Continue `uop/*` integration: complete remaining `UOp` methods, symbolic/decomposition
   coverage, `spec.py`, and `validate.py` against the upstream source layout.
3. Continue Tensor mixin integration: full dtype/bitcast coverage, remaining random APIs, full
   lazy indexing, higher ops, complete conv/pool options, and exact public method semantics.
4. Integrate schedule/render/codegen into one executable backend path.
5. Add package-level differential tests against real tinygrad for each integrated subsystem.
6. Only then attempt real examples such as `examples/beautiful_mnist.py`.
