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
  - integrated `tinygrad-rss/src`: 123 RSS files, 43,719 LOC.
  - vendored `tinygrad-rss/vendor/tinygrad/runtime/autogen`: 88 generated Python files,
    exactly copied from upstream tinygrad commit `fa400f9790ab9a684387b02e958658217b33e7c1`.
  - standalone `port-rss`: 55 RSS files, about 12.1k LOC.
  - rough source coverage inventory:
  - command: `python3 tools/port_coverage.py --limit 8`
  - result: `tensor.py` 106/106 symbols, `mixin/__init__.py` 82/82 symbols,
    `uop/ops.py` 221/221 symbols; 409/409 total rough symbols covered.
  - full handwritten Python inventory, excluding `runtime/autogen`: 118/118 upstream Python
    source files now have matching RSS source paths, and the rough symbol inventory reports
    2628/2628 covered. This is a map/inventory milestone only; it does not prove exact tinygrad
    semantics for every symbol.
  - focused dtype/helpers/source-map compatibility batch: `dtype.py`, `helpers.py`,
    `uop/__init__.py`, the source-path-only `mixin/*` modules, `function.py`,
    `runtime/graph/metal.py`, `runtime/ops_hip.py`, `runtime/ops_npy.py`, and
    `runtime/support/compiler_qcom.py`: source-path mapping is complete and the rough symbol
    inventory is 100%. Python runtime-adjacent helpers such as env/diskcache/fetch/profile/tqdm
    are deterministic RSS facades, not full Python runtime behavior.
  - focused function/mixin semantic wrapper batch: `function.py`, `mixin/creation.py`,
    `mixin/dtype.py`, `mixin/elementwise.py`, `mixin/movement.py`, `mixin/rand.py`, and
    `mixin/reduce.py` now expose source-path-local wrappers over the integrated tensor call,
    creation, dtype, elementwise, movement, random, and reduction helpers. These wrappers are
    exercised by the active smoke; they are still RSS graph-helper entry points, not Python
    descriptor/decorator object behavior.
  - focused mixin source-surface expansion batch: the same mixin files now expose grouped wrappers
    for upstream-style const/empty/same-dtype creation, dtype predicates/cast, elementwise ALU,
    comparison, bitwise, unary, contiguous/detach helpers, movement shape/view/pad/squeeze/
    transpose/flatten/cat/stack helpers, direct RNG constructors, and reduce/stat/softmax/arg
    helpers over the integrated tensor layer. These are smoke-tested wrappers over existing tensor
    semantics, not a Python method descriptor/object model.
  - focused runtime backend wrapper batch: `runtime/graph/metal.py`, `runtime/ops_hip.py`,
    `runtime/ops_npy.py`, and `runtime/support/compiler_qcom.py` now expose deterministic
    source-path-local state wrappers for Metal graph launch metadata, HIP/NPY device allocation
    and transfer shells, and QCOM compiler/check/disassembly hooks. These are smoke-tested RSS
    facades over existing memory/compiler helpers, not real Metal ICB, HIP driver, NumPy device,
    or QCOM LLVM/compiler behavior.
  - focused compiler/renderer batch: `codegen/opt/*` 54/54 symbols and
    `renderer/cstyle.py` 40/40 symbols; 94/94 total rough symbols covered across
    the current optimizer/renderer audit set.
  - focused runtime-engine/device/codegen batch: `engine/jit.py`, `engine/realize.py`,
    `device.py`, `renderer/cstyle.py`, and `codegen/opt/*`: 210/210 rough symbols covered.
  - focused Tensor mixin/function batch: `mixin/creation.py`, `mixin/dtype.py`,
    `mixin/elementwise.py`, `mixin/movement.py`, `mixin/rand.py`, `mixin/reduce.py`,
    and `function.py`: 194/194 rough symbols covered.
  - focused nn batch: `nn/__init__.py`, `nn/optim.py`, `nn/state.py`, and `nn/datasets.py`:
    68/68 rough symbols covered. This is source-surface coverage with lightweight RSS facades,
    not full training semantics.
  - focused NN layer-surface batch: `nn/__init__.py` now has source-shaped state/call helpers for
    BatchNorm/BatchNorm2d/BatchNorm3d, Conv1d calls, Conv2d, ConvTranspose2d, Linear, LayerNorm,
    LayerNorm2d, GroupNorm, InstanceNorm, RMSNorm, and Embedding over the integrated tensor layer.
    These are deterministic tensor-graph layer helpers with active smoke coverage, not full Python
    class/descriptors, training mode globals, or optimizer-fused module semantics.
  - focused NN optimizer semantics batch: `nn/optim.py` now carries optimizer kind,
    hyperparameters, and per-parameter momentum/Adam state buffers for SGD, Muon, Adam, and AdamW.
    `_step` computes graph-level weight decay, momentum/Nesterov, Muon update routing, and
    Adam/AdamW moments plus bias correction. The active smoke exercises these state updates; this
    is still not fused optimizer scheduling, training globals, device movement, LARS/LAMB trust
    ratios, or exact tinygrad optimizer execution.
  - focused NN state-file batch: `nn/state.py` now parses simple in-memory safetensors metadata
    from tensor-backed bytes using the upstream 8-byte header length, JSON object keys, dtype,
    shape, and `data_offsets`; `safe_load` slices raw payload bytes into typed/shaped tensors.
    It also extracts uncompressed tar regular-file members into tensor-backed byte payloads with
    upstream-style 512-byte record stepping and directory skipping. The active smoke loads a
    minimal `U8[4]` safetensors payload and a synthetic tar containing two regular files plus a
    directory entry. Zip/torch pickle loading, compressed archives, file-name wrappers, external
    storage, and full model object traversal are still not exact upstream behavior.
  - focused renderer/runtime/nn backend batch: `nn/*`, `renderer/llvmir.py`, `renderer/ptx.py`,
    `renderer/wgsl.py`, `renderer/nir.py`, `runtime/ops_python.py`, `runtime/ops_null.py`,
    `runtime/ops_cpu.py`, and `runtime/ops_disk.py`: 201/201 rough symbols covered.
    The disk runtime facade now follows upstream deterministic lifecycle/view metadata more
    closely: reopening with a smaller size preserves the original mapping size and increments
    refcount, larger reopen is exposed as a checked failure, close only releases on the final
    refcount, `_offset` uses absolute disk offsets, and `_copyout_sharded` emits page-aligned
    shard metadata with first-shard minor offsets. This is still not real mmap or io_uring IO.
  - focused runtime support batch: `runtime/support/hcq.py`, `runtime/support/memory.py`,
    `runtime/support/elf.py`, `runtime/support/compiler_cpu.py`, `runtime/support/c.py`, and
    `runtime/graph/hcq.py`: 167/167 rough symbols covered.
  - focused backend runtime support batch: `runtime/support/am/ip.py`,
    `runtime/support/am/amdev.py`, `runtime/support/amd.py`, `runtime/support/nv/ip.py`,
    `runtime/support/nv/nvdev.py`, `runtime/support/usb.py`, `runtime/support/system.py`,
    `runtime/support/autogen.py`, and `runtime/support/mlx/mlxdev.py`: 334/334 rough symbols
    covered. This is source-shaped backend/device support coverage with deterministic smoke
    execution, not real kernel-mode IO, firmware loading, VFIO, USB transport, or GSP/SMU control.
  - focused compiler/graph/support marker batch: `runtime/graph/cuda.py`,
    `runtime/graph/metal.py`, `renderer/amd/generate.py`, `runtime/support/compiler_amd.py`,
    `runtime/support/compiler_cuda.py`, `runtime/support/compiler_mesa.py`,
    `runtime/support/compiler_qcom.py`, `runtime/support/objc.py`, package-marker
    `__init__.py` modules, `nn/torch.py`, and `viz/__init__.py`: 80/80 rough symbols covered.
    This is source-shaped compiler/disassembly/generator/runtime facade coverage with
    deterministic smoke execution, not real NVRTC, HIP/COMGR, Mesa/NIR, Objective-C runtime,
    CUDA graph, or AMD ISA XML/PDF generation behavior.
  - focused renderer ISA/AMD batch: `renderer/amd/dsl.py`, `renderer/amd/elf.py`,
    `renderer/amd/sqtt.py`, `renderer/isa/x86.py`, `renderer/__init__.py`,
    `renderer/amd/__init__.py`, and `renderer/isa/__init__.py`: 135/135 rough symbols covered.
  - focused target runtime batch: `runtime/ops_amd.py`, `runtime/ops_nv.py`,
    `runtime/ops_cuda.py`, `runtime/ops_metal.py`, `runtime/ops_webgpu.py`,
    `runtime/ops_qcom.py`, `runtime/ops_dsp.py`, `runtime/ops_rdma.py`,
    `runtime/ops_tinyfs.py`, and `runtime/ops_cl.py`: 331/331 rough symbols covered.
  - focused graph-transform batch: `callify.py` and `gradient.py`: 17/17 rough symbols covered.
  - focused ONNX runner execution batch: `nn/onnx.py`: 41/41 rough symbols covered. The RSS port
    now has runner value-table helpers and graph execution for Add/Sub/Mul/Div, Relu, MatMul,
    Reshape, Transpose, plus simple Gemm and Conv paths over the tensor helpers. Active smoke
    checks real tensor values for Add, Relu, Reshape, Transpose, MatMul, Gemm, and Conv. This is
    still not full parsed-model execution parity: attribute decoding, opset dispatch,
    optional/sequence types, control flow, quantization, pooling, broadcast edge cases, and full
    reshape `allowzero` behavior still need deeper work.
  - focused LLM facade batch: `llm/gguf.py`, `llm/model.py`, and `llm/cli.py`: 54/54 rough
    symbols covered. This is a source-shaped GGUF/tokenizer/model/server facade with tiny smoke
    execution; real GGUF quantized tensor decoding and full transformer execution still require
    deeper tensor/runtime work.
  - focused viz facade batch: `viz/cli.py` and `viz/serve.py`: 49/49 rough symbols covered. This is
    source-shaped profile formatting/server/profile-layout/static-analyzer surface coverage with
    tiny smoke execution, not the full browser UI/server/runtime trace stack.
  - current focused audit slice: `llm/*`, `viz/cli.py`, `viz/serve.py`, `callify.py`,
    `gradient.py`, `runtime/ops_hip.py`, `runtime/ops_npy.py`, and `nn/onnx.py`: 176/176 rough
    symbols covered (100.0%).
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
- RSScript now exposes `Bytes.to_string` across stdlib interfaces, runtime ABI, reg VM, and Rust
  runtime so tensor-backed byte buffers can feed JSON parsers for safetensors-style metadata.
- RSScript now exposes `HttpResponse.bytes` across stdlib interfaces, runtime ABI, reg VM, and
  Rust runtime so URL-backed tensor creation can consume response bodies as bytes instead of
  lossy text.
- RSScript sync `Http.get` now drives the existing reqwest/tokio implementation through the
  native pending executor and preserves raw response bytes alongside text, so current
  `from_url` can execute real byte downloads when network access is available.
- RSScript Rust lowering now emits Rust `.clone()` for checker-supported builtin value types
  (`List`, `Bytes`, and `Buffer`) instead of leaking unresolved `Type_clone` helper symbols.
- RSScript Rust lowering now raw-escapes Rust reserved/future-reserved identifiers, including
  Rust 2024's `gen`, so source-shaped tinygrad support/autogen helpers can keep upstream names.

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
  `track_rewrites`, and `profile_matches`, plus source-shaped compile/render helper names
  `_get_clause`, `do_process_and`, `wrap`, `_final_render`, `_get_code`, and `upat_compile`
  over the current pattern metadata; Python callable bytecode reconstruction, dynamic `exec`-style
  UPat compilation, and trace/profile collection are still not represented.
- `uop/spec.rss`: first integrated interned-graph verifier for the shared core currently built by
  the package: CONST/SPECIAL/RANGE, `DEFINE_VAR`/`BIND`, `PARAM`, `DEFINE_LOCAL`/`DEFINE_REG`,
  upstream-shaped device/buffer/copy/multi-device graph nodes, ALU dtype rules, same-itemsize `BITCAST`, WHERE/MULACC, buffers, movement
  nodes, STACK/SINK/MSTACK, source-shaped program/call/stage/slice/codegen-facing nodes,
  codegen helper `CUSTOM`/`CUSTOMI`, `INS`, `GETADDR`, `WMMA`/`SHAPED_WMMA`, and `VCAT`/`PTRCAT` nodes,
  `UNROLL`/`CONTRACT` dtype-count/product validation including void `CONTRACT` for store contraction,
  tuple/gettuple, index/load-store, source-shaped `REDUCE`/`STORE` range tails,
  control-flow `IF`/`ENDIF`, order nodes including `END` range validation, symbolic `STACK`
  movement shape args, recursive source validation, and source-shaped verifier/render test entry
  points `validate_index`, `type_verify`, `eval_pyrender`, and `test_pyrender`.
- `uop/validate.rss`: first integrated source-shaped `tinygrad/uop/validate.py` slice for masked
  index bounds validation over the interval subset already covered by `uop_min_max`: static false
  gates are accepted, active/unknown gates require the index interval to be fully inside
  `[0, size)`, `INDEX` nodes can be validated by deriving flat buffer size from the base shape
  and reading the index/gate sources, and the local `validate_index_with_z3` entry point aliases
  this conservative interval proof. It also exposes source-shaped integer-model helpers
  `z3_cdiv`, `z3_floordiv`, `z3_xor`, `create_bounded`, and `uops_to_z3` for current concrete
  interval summaries. Full Z3-backed boolean/arithmetic solving remains unported.
- `uop/render.rss`: first integrated source-shaped `tinygrad/uop/render.py` slice over interned
  UOp ids, covering the core symbolic renderer (`DEFINE_VAR`, named `SPECIAL`, `PARAM`, `RANGE`, `CONST`,
  `CAST`, `BIND`, unary/binary/ternary ALU render rules, `INDEX`/`STAGE`, and `STACK`),
  inference-specific rendering for div/mod and `BITCAST`, compact UOp line printing, and a first
  `pyrender`-style reconstruction slice for constants, variables, params, casts/bitcasts, binary
  ALU expressions, `WHERE`, and `INDEX`. Source-shaped render entry points now cover
  `pretty_print`, `print_uops`, `strip_binary_parens`, `srcs`, `render_marg`,
  `_render_with_splits`, and `pyrender` over the current UOp-id model; the split-depth assignment
  planner and full upstream pyrender call/function handling remain incomplete.
- `uop/divandmod.rss`: first integrated source-shaped `tinygrad/uop/divandmod.py` slice over
  interned UOp ids, covering interval cancellation, nested `(x%(k*c))//c` and `%c` rewrites,
  remove-nested-mod-in-sum, binary numerator folding, gcd-with-remainder factoring,
  nest-by-factor splitting, and factor-remainder splitting for non-negative integer index
  expressions.
- `uop/decompositions.rss`: first integrated source-shaped `tinygrad/uop/decompositions.py` late
  rewrite slice over interned UOp ids, covering Python `FLOORDIV`/`FLOORMOD` lowering to truncating
  `CDIV`/`CMOD` with mixed-sign adjustment, `MAX` lowering to `CMPLT`+`WHERE`, and
  `RECIPROCAL`/multiply-by-reciprocal lowering to `FDIV`, plus power-of-two floor-mod to `AND`,
  power-of-two multiply/divide to shifts, `x*-1` to `NEG`, `x+(-y)` to `SUB`, and
  multiply/shift-plus-add to `MULACC`, non-negative `CMOD` to `x-d*CDIV(x,d)`, selected signed
  comparison canonicalizations, tight integer range to `CMPEQ`, and logical-not-of-`CMPNE` to
  `CMPEQ`, plus `THREEFRY(uint64,uint64)` lowering to primitive uint32/uint64 add/xor/shift/cast/or
  nodes for RNG support. Source-shaped helper names from upstream are now exposed for the current
  UOp-id model: dtype bit helpers (`mantissa_bits`, `exponent_bias`, `exponent_mask`), shift/round/
  exponent helpers (`shr`, `shl`, `rintk`, `pow2if`, `ilogb2k`, `ldexp3k`, `ldexp2k`, `frexp`),
  first reduction/transcendental scaffolding (`payne_hanek_reduction`, `cody_waite_reduction`,
  `trig_poly`, `sin_poly`, `_ifand`, `sin_poly_small`, `sin_poly_large`, `xsin`, `xexp2`, `xlog2`,
  `xpow`), integer division/RNG helpers (`magicgu`, `fast_idiv`, `threefry2x32`), long/float
  scaffolding (`unpack32`, `reindex`, `l2i`, `rne`, `f2f`, `f2f_clamp`, `f2f_load`, `f2f_store`),
  and source-shaped rewrite entry points (`get_transcendental_patterns`, `floordiv_to_idiv`,
  `floormod_to_mod`, `get_late_rewrite_patterns`, `do_dtype_decomps`). The source-shaped entry
  points delegate to implemented late rewrites where available; the large transcendental
  approximations, exact long-integer decomposition, exact float-format conversion, broader
  boolean/comparison normalization, and target-op-availability rewrite machinery remain partial or
  unported.
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
  Source-shaped public renderer names now cover non-native float pattern hooks, bf16 cast
  construction, dtype inventory, WMMA metadata collection, `render_dtype`, `render_cast`,
  `_render`/`render`/`render_kernel`, vector typedef prefixes, render body/entry helpers,
  supported dtype sets, OpenCL `aux`, CUDA/HIP fp8/OCML helpers, CDNA arch predicates, and an
  assembly facade for HIP. It still lacks upstream's full inline heuristics, exact target-specific
  parameter effects, real target compiler objects, full WMMA prefix emission, and schedule integration.
- `renderer/llvmir.rss`, `renderer/ptx.rss`, `renderer/wgsl.rss`, and `renderer/nir.rss`: first
  source-shaped backend renderer facade batch. The integrated names cover LLVM dtype/constant/cast
  spelling and WMMA/footer/function shells; PTX value, memory-space, WMMA, and cast-modifier
  helpers; WGSL packed load/store/rewrite support helpers and buffer type spelling; and NIR helper
  names for Mesa symbol lookup, def/source/type text, ALU/cast/channel/immediate/index/vector/image
  facades, parameter/prerender/postrender shells, and float-type spelling. These are deterministic
  RSS facades over the current UOp graph and string metadata, not real LLVM/PTX/WGSL/NIR backend
  emitters or native Mesa/LLVM/CUDA compiler integrations yet.
- `renderer/__init__.rss`, `renderer/isa/__init__.rss`, `renderer/isa/x86.rss`,
  `renderer/amd/__init__.rss`, `renderer/amd/dsl.rss`, `renderer/amd/elf.rss`, and
  `renderer/amd/sqtt.rss`: first source-shaped renderer ISA/AMD facade batch. The integrated
  names cover renderer program metadata, ISA register/instruction helpers, x86 instruction and
  operand utility names, AMD DSL register/bitfield/instruction metadata, AMD format detection,
  linear assembly byte shells, and SQTT packet/decode/map/format helper names. These are
  deterministic state/string/byte facades, not exact x86 lowering, AMD instruction table parity,
  or real machine-code assembly/disassembly yet.
- `runtime/ops_amd.rss`, `runtime/ops_nv.rss`, `runtime/ops_cuda.rss`,
  `runtime/ops_metal.rss`, `runtime/ops_webgpu.rss`, `runtime/ops_qcom.rss`,
  `runtime/ops_dsp.rss`, `runtime/ops_rdma.rss`, `runtime/ops_tinyfs.rss`, and
  `runtime/ops_cl.rss`: first source-shaped target runtime facade batch. The integrated names
  cover CUDA argument/time/synchronization helpers, OpenCL status checks, Metal/WebGPU buffer and
  command helpers, AMD/NV/QCOM packet/register/allocation/profiling helper surfaces, DSP RPC/lib
  helpers, RDMA work-queue encoding, and tinyfs connection/copy shells. These are deterministic
  metadata/state facades over RSS structs, not real CUDA/Metal/WebGPU/OpenCL/AMD/NV/QCOM/DSP/RDMA
  driver execution yet.
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
  and is idempotent when the program already has a `LINEAR` child. Source-shaped `do_split_ends`
  now delegates to the integrated `pm_split_ends` helper.
  Pre-existing IF rejection, ISA register allocation, compile/binary, and full schedule/device
  rewrite orchestration remain unported.
- `codegen/late/regalloc.rss`: first source-shaped boundary slice for
  `tinygrad/codegen/late/regalloc.py`, covering upstream's pseudo-op set
  (`CONST`, `NOOP`, `AFTER`, `BARRIER`, `GROUP`) and an explicit capability predicate that keeps
  real linear-scan allocation disabled until the RSS IR carries ISA `Register` tags from a target
  renderer. Source-shaped `vdef` and `regalloc_rewrite` entry points are present as current-IR
  metadata/passthrough helpers, not a real ISA register allocator.
- `codegen/late/gater.rss`: integrated source-shaped `tinygrad/codegen/late/gater.py`
  slice, covering the `pm_move_gates_from_index` rewrites for 1D and 2D
  `INDEX(WHERE(gate, idx, INVALID))` nodes feeding `LOAD` or `STORE`, including optional
  pointer casts. The port rebuilds the index with the raw idx values, moves the validity
  condition onto the load/store gate, supplies a zero-like load fallback, combines an existing
  store gate with the moved index gate, and folds `WHERE` around gated loads into the load alt
  value for direct and inverted gate forms.
- `codegen/late/expander.rss`: integrated source-shaped `tinygrad/codegen/late/expander.py`
  slice, covering the upstream entry point names `_expand_arg_to_idx`, `_choices_from_args`,
  `_swizzle_args`, `do_expand`, `do_contract`, `end_unrolls`, `fix_reduce_unroll`,
  `fix_store_unroll`, and `fix_group_for_reduce`. It ports axis-size
  choice/index helpers and the compact normalization rewrites for
  `CONTRACT`, empty/double `UNROLL`, and `END` consuming `UNROLL` axes. It also has the common
  same/mixed-axis `do_expand` path for elementwise, memory, ordering, and `WMMA` roots, including `UNROLL` source stripping,
  mixed-axis `GEP` swizzles, scalar broadcast, vector `VCAT` repetition, range-arg passthrough,
  and `GEP` arg expansion. It also has the first `pm_pre_expander` slice: `UNROLL`/`UPCAST`
  ranges become `UNROLL` constants, and `REDUCE`/`STORE` nodes carrying `UNROLL` tails are
  contracted before later expansion. It also has the first BufferizeOpts-local
  `GROUP_REDUCE` rewrite: grouped reduce ranges are converted to local `STAGE` storage and
  final `AXIS_REDUCE` loops while preserving upstream local ranges. Broadcast `STACK`s now push
  through `AFTER`/`END`, two-UNROLL `STAGE` inputs now contract before expansion, and normalized
  REG value-index roots scalarize to per-lane `INDEX` nodes. Exact upstream WMMA excluded-axis
  metadata remains approximated by the current flat RSS WMMA arg payload.
- `codegen/late/devectorizer.rss`: integrated source-shaped
  `tinygrad/codegen/late/devectorizer.py` slice, exposing the upstream names
  `_drop_valid_stmts`, `simplify_valid_load`, `simplify_valid_image_load`, `expand_index`,
  `fold_expanded_index`, `cat_after_store`, `gep_on_store`, `split_load_store`,
  `get_image_idx`, `image_fixup`, `no_vectorized_wmma`, `no_vectorized_alu`,
  `no_vectorized_buf`, `no_vectorized_index`, `horizontal_reduce`, `reduce_to_acc`,
  `merge_reduce_ends`, `add_load`, and `make_image` over the cache-backed RSS IR. It covers the
  `devectorize_alu` scalarization
  rule for vector ALU, `CAST`, `BITCAST`, and the local flat-metadata `WMMA` shape: vector operations are rebuilt as scalar lane
  operations over `GEP` sources and wrapped in a `STACK`, plus the upstream
  no-range horizontal `REDUCE` lowering for `ADD`, `MUL`, and `MAX`. It also has the upstream
  `CAST(AFTER(x, deps...)) -> AFTER(CAST(x), deps...)` ordering rewrite. It also has the first `pm_render`
  normalization rules: vector `CONST` nodes become scalar-constant `STACK`s, multi-lane `GEP`
  becomes a `STACK` of lane `GEP`s, `GEP(x, 0)` on scalar `x` unwraps, and one-element `STACK`
  unwraps. The context-free `load_store_folding` rules for `INDEX(STACK(bufs), vec_idx)`,
  `GEP` after `LOAD`, `GEP` on `STORE`, `PTRCAT` after `LOAD`, and `PTRCAT` after `STORE`
  are also integrated, including vector load/store splitting through cast-backed `INDEX`
  pointers while preserving load extras and store gates, along with the `pm_add_loads` rules that insert `LOAD` for
  non-pointer `INDEX` nodes and clean nested `LOAD(LOAD(ptr))` and `STORE(LOAD(ptr), value)`.
  Source-shaped facades also include conservative folded-index/image entry points, scalar
  DEFINE_LOCAL/DEFINE_REG buffer devectorization, addrspace-aware `BUFFER(ParamArg)` local/reg
  buffer devectorization, simple CAST-backed index lane expansion,
  range-backed `reduce_to_acc` lowering through REG accumulator init/update/end/readback, and
  tensor-core `WMMA + addend` accumulator folding. Full ImageDType
  handling, exact folded-index regrouping, and exact mergeable-END consolidation remain later
  devectorizer slices. The REG/local buffer devectorization slice now also covers upstream-shaped
  `CAST(...).index`, broadcast-index, and GEP-index lane mapping for addrspace-tagged REG buffers.
- `codegen/__init__.rss`: first integrated source-shaped `tinygrad/codegen/__init__.py` slice,
  wiring `PROGRAM(SINK, DEVICE, LINEAR)` through the first `do_estimates` equivalent and CStyle
  renderer. It computes integer upper-bound `Estimates.from_uops(..., ignore_indexing=True)`-style
  metadata for the current interned UOps (`ops`, load/store bytes, and unique capped buffer memory),
  derives first `ProgramInfo.from_sink`-style program metadata (`vars`, `globals`, `outs`, `ins`,
  and integer `SPECIAL` launch dimensions for `g*`, `l*`, and `i*` names) for the current lowered
  sink shape, then appends a `SOURCE` child with the rendered kernel text. Source-shaped entry
  points now cover `full_rewrite_to_sink`, including upstream-shaped `DEFINE_LOCAL`/`DEFINE_REG`
  normalization into addrspace-tagged `BUFFER(ParamArg)` nodes and register-load removal,
  `line_rewrite`, `do_linearize`, `do_estimates`,
  `do_assemble`, `do_render`, `do_compile`, `do_to_program`, and `to_program` for the supported
  C-style path. It also has the first
  host compile/syntax bridge for rendered CStyle source, writing the attached `SOURCE` to a path
  and invoking `clang -x c -std=c11 -fsyntax-only` through RSScript `Path`/`Process`, returning
  source id/length, exit status, stderr length, and ok/fail metadata. It also has the first hosted
  executable C harness bridge for the current generated kernel shape: combine `SOURCE` with a
  supplied C `main`, compile with `clang`, run the executable through `Process`, and capture stdout
  plus compile/run status. The first `to_program`-style orchestration for the supported C-style path is:
  `do_linearize -> do_estimates -> do_render`. Full symbolic `sint` estimates,
  aux metadata, real compiler-backed `BINARY` materialization, general buffer-backed runtime invocation,
  and full schedule/device rewrite orchestration remain unported.
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
  Source-shaped entry points now cover `_unwrap_src`, `_split_after`, `create_schedule`,
  `create_new_buffer`, `lower_sink_to_linear`, and `create_linear_with_vars` over this current
  scheduler model.
- `schedule/memory.rss`: first integrated source-shaped `tinygrad/schedule/memory.py` slice:
  buffer collection through `BUFFER`/`MSELECT`/`MSTACK`, held-buffer and disk/tinyfs rejection,
  first/last lifetime tracking, copy-vs-compute lane separation, per-device/per-lane int8 arenas,
  first-fit lifetime reuse for expired same-lane allocations, and buffer-to-`SLICE` rewrite for
  planned `LINEAR` calls. Exact TLSF split/coalesce behavior and upstream allocator edge cases
  remain unported. Source-shaped `_collect_bufs` and `_can_plan` delegate to the integrated planner
  checks.
- `schedule/indexing.rss`: integrated source-shaped `tinygrad/schedule/indexing.py` slice:
  exposes the upstream entry point names `realize`, `realize_srcs`, `realize_store_after_src`,
  `new_range`, `create_bufferize_and_index_based_on_ranges`,
  `convert_pad_to_where_to_keep_behavior_local`, `convert_reduce_to_reduce_with_ranges`,
  `remove_movement_op_after_rangeify`, `_apply_reshape`, `apply_movement_op`,
  `run_rangeify`, and `render_ranges` over the cache-backed RSS IR. It covers
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
- `schedule/rangeify.rss`: first source-shaped `tinygrad/schedule/rangeify.py` facade over the
  cache-backed RSS IR, exposing `add_ranges_to_store`, `lower_shaped_wmma`, `found_after`,
  `_mop_index`, `fix_store_hazard`, `split_reduceop`, `resolve_function`,
  `cleanup_dead_axes`, `gate_substitute`, `remove_bufferize`, `remove_noop_bufferize`,
  `late_buffer_view`, `limit_bufs`, `bufferize_to_store`, `flatten_bufferize`, `debuf`,
  `unbind_kernel`, `handle_after`, `renumber_range`, `find_bufs`, `get_contiguous`,
  `split_store`, and `get_kernel_graph`. It now composes `get_kernel_graph` through the integrated
  indexing rangeify analysis/rewrite, device buffer limiting, stage/buffer cleanup, and split-store
  passes. `split_store` rewrites kernel bodies by replacing buffer-like inputs with PARAMs,
  unwrapping BIND/AFTER/CONTIGUOUS/NOOP nodes, renumbering ranges, stripping local STAGE devices,
  and returning a `CALL(SINK(...), original buffers...)` shape for normal stores. The facade also
  implements practical movement-index lowering, store range injection, simple hazard contiguity,
  noop/dead-axis staging cleanup with shrink/reshape/expand preservation, disk/tinyfs late buffer
  views as `SLICE`, global stage-to-buffer store lowering
  through `AFTER(buffer, END(STORE(INDEX(buffer,...), value)))`, reuse of existing `AFTER` buffers,
  local stage-to-`DEFINE_LOCAL` lowering with `BARRIER`,
  buffer-to-param placeholder conversion, range renumbering, and the first upstream
  `pm_syntactic_sugar` cleanup for nested pointer `INDEX` flattening plus elementwise/const
  `INDEX` pushdown before range analysis, plus removable `STAGE` range substitution for
  `remove_bufferize` and source-shaped `SHAPED_WMMA` lowering into upcast-indexed
  `CONTRACT` inputs, `WMMA`, per-lane `GEP`, and a register `AFTER(STORE...)` result. It also
  resolves `FUNCTION` bodies by gathering PARAM slots and substituting provided arguments with
  dtype/shape/axis checks, ports the default large-reduction split rewrite into
  reshape/permute/reduce/contiguous/reduce/reshape form, and implements the first device buffer
  cap rewrite for METAL/WEBGPU by staging elementwise children through fresh LOOP ranges when
  buffer-like inputs exceed the backend cap. `bufferize_to_store` now reuses `AFTER` buffers with
  upstream-shaped writeback repair for nested `INDEX(STAGE(INDEX(...)), ...)` targets, drops
  self-stores after unwrapping, and ends repaired stores over the union of target and consumer
  ranges. Dead-axis cleanup now follows the upstream guards for always-run/`AFTER` sources and
  removes unused `RANGE` axes as well as constant axes; store hazard repair now forces contiguity
  only when the self-referencing source also contains unsafe movement. `flatten_bufferize` now computes the single flattened
  index through the upstream RESHAPE movement mapping and reshapes the staged result back to the
  original staged shape. `remove_bufferize` now carries the upstream conservative cost gates for
  expressions touching more than three accessed buffers and for reduce subtrees that read
  bufferized/param data. A first post-split WAR assign repair now appends reader `AFTER` deps to
  later writer assigns when a split kernel reads a buffer written by another assign. `split_store`
  now preserves `COPY`/`SLICE` as special call bodies with ended ranges instead of wrapping them in
  `SINK`, and debuf now emits the upstream max-shape param plus shrink-back path for symbolic
  shapes. Assign repair now carries the upstream circular-dependency predicate and avoids adding
  the dependency edge that would create a cycle; exact Python-style exception propagation for this
  diagnostic remains pending. `remove_bufferize` now carries the upstream `PCONTIG > 2`
  relaxations for high-buffer-count expressions and reduce-backed partial-contiguous local
  re-bufferization, and `limit_bufs` now honors the upstream `MAX_KERNEL_BUFFERS` override before
  falling back to device defaults. Richer symbolic max-shape inference and full runtime env plumbing
  remain intentionally unported.
  The first `pm_const_buffer_folding`/`pm_add_buffers` cleanup batch is also present: const
  stages, const indexes, const copies, const-backed `MSTACK` indexes, `NOOP(CONST)`,
  self-stores, `END(NOOP)`, invalid writes, `AFTER(..., NOOP)`, reshape-through-`MSELECT`/
  `MSTACK`, and reshape argument stripping on `CALL`.
  It also has `split_store` call wrappers that
  expose store buffer arguments to the scheduler. Exact upstream kernel graph repair, buffer-cost
  modeling beyond these guards, full runtime env propagation, and exact exception propagation remain
  later rangeify slices.
- `schedule/allreduce.rss`: first integrated source-shaped `tinygrad/schedule/allreduce.py` slice:
  naive allreduce graph construction for supported multi-device `MULTI`/`MSTACK` inputs by
  normalizing the input through `CONTIGUOUS`, selecting/copying each shard to the target device,
  and reducing copied shards with `ADD`/`MUL`/`MAX`. It also has first source-shaped chunked
  all2all/ring graph builders: flatten/chunk shards, reduce copied chunks, gather to the target,
  pad chunks back into place, and reshape to the original shape. Source-shaped `handle_allreduce`
  still delegates to the naive path until upstream `RING`/`ALL2ALL` knobs, thresholds, and exact ring
  source-copy metadata are wired through; `create_allreduce_function` provides the current
  cache-backed store/after wrapper while full Python `call(..., precompile=True)` lowering remains
  tied to later function-call scheduler/runtime work.
- `codegen/gpudims.rss`: first integrated source-shaped `tinygrad/codegen/gpudims.py` slice for
  the current all-integer RSS IR: `_dim_max`, `_group_dims`, `_split_dims`, `get_grouped_dims`,
  and `add_gpudims`. It covers max-size grouping, factor splitting, grouped `SPECIAL` index
  creation, flat-to-coordinate reconstruction, and conservative `RANGE -> SPECIAL` substitution
  for `GLOBAL`/`THREAD` and `WARP`/`LOCAL`/`GROUP_REDUCE` axes. It also ports the upstream
  missing-local global-store guard: if a global `STORE` index omits local ranges used by the
  kernel, `add_gpudims` wraps the index in `WHERE(local_idx == 0, idx, INVALID)` so the existing
  gate-from-index pass can lower it to a gated store. The CStyle `PROGRAM(SINK, DEVICE)` path now
  applies `pm_add_gpudims` and graph-wide gate-from-index rewriting before linearization, so those
  guarded indexes become `IF`/`STORE`/`ENDIF` line streams before rendering. Symbolic `sint.vmax`,
  renderer product-limit tuning, and broader upstream rewrite-pipeline parity remain future fidelity
  work.
- `codegen/simplify.rss`: first integrated source-shaped `tinygrad/codegen/simplify.py` facade:
  `flatten_range`, `count_divmod`, `simplify_merge_adjacent`, `mark_gated`, `mark_range_mod`,
  `do_substitute`, `no_range`, `reduce_unparented`, `reduce_collapse`, `reduce_load_collapse`,
  and `no_load`. It uses existing `graph_rewrite`, range discovery, and `memory_replace_many`
  primitives, with a real `reduce_unparented` path for unused ADD/MUL reduce ranges and
  conservative collapse hooks for pattern-heavy upstream cases. `mark_gated` now recognizes
  `WHERE(valid, idx, INVALID)` index operands, extracts `range < const` clauses from `AND`-split
  validity guards, and substitutes guarded ranges with the tighter bound while leaving ungated
  index ranges at their full end. The split-range pass now uses the upstream divisibility rule for
  `RANGE % const` and substitutes splittable ranges with `hi * const + lo` replacement ranges that
  preserve the original range axis metadata plus the upstream high/low marker axis. The first
  source-shaped load-index collapse is integrated for the upstream gated-ADD pattern
  `REDUCE(WHERE(idx != range, 0, expr), range)`, replacing the reduced range in the payload with a
  valid bounded index and wrapping the result in the same bounds guard. The range-bound
  reduce-collapse family now covers the upstream upper-bound, lower-bound, and two-sided window
  folds for range-free values, clamping counts to the reduced range extent before multiplying by the
  payload. The first arithmetic comparison lift batch is also integrated: `(x + y) < c` and
  `(x * y) < c` feed range-bound collapse when the lifted operand and bound are range-free, and
  `(x + y) != c` feeds load-index collapse. The next PatternMatcher parity batch covers
  REDUCE-on-ADD distribution, DEFINE_VAR-gated AND-on-WHERE extraction, and MUL-casted-bool
  normalization into `WHERE(gate, x, 0)` for the supported reduce-collapse path. Full parity for
  the generic reduce-collapse driver, broader symbolic composition, and exact placeholder
  substitution remains later fidelity work.
- `codegen/opt/__init__.rss`, `codegen/opt/postrange.rss`, `codegen/opt/tc.rss`,
  `codegen/opt/heuristic.rss`, and `codegen/opt/search.rss`: first integrated source-shaped
  `tinygrad/codegen/opt` batch. `OptOps`, `Opt`, and `check` are present, and
  `PostrangeScheduler` exposes range inventory, shape strings, output/globalizable ranges,
  conservative loop-to-global conversion, range splitting through `shift_to`, `apply_opt`,
  reduce/index discovery, output-shape metadata, and `apply_opts` over the current RSS UOp IR.
  `TensorCore` metadata now covers dims, thread counts, element counts, dtype pairs, opts,
  local/upcast/reduce axes, swizzle remaps, base shape strings, permute derivation, string
  rendering, constructor validation, and CUDA/AMD arch-table facades over a representative
  source-shaped subset. `hand_coded_optimizations` applies conservative group/upcast/unroll/local
  batches over the current scheduler, and `search.rss` exposes source-shaped global-size clamping,
  worker/timeout/compile/buffer-allocation facades, action generation, and deterministic beam
  selection over legal `PostrangeScheduler` variants. Tensor-core lowering, full arch table
  breadth, real device timing/compile caching for beam search, renderer-specific heuristics, and
  exact upstream symbolic range semantics remain later fidelity work.
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
  tuple-device `ALLREDUCE(ADD)`, and symbolic `SHRINK`. Source-shaped aliases now cover
  `mstack_early_shrink`, `alu_multi`, `reduce_multi`, movement multi rewrites, `copy_multi`,
  `store_after_multi`, `passthrough_multi`, `rewrite_into_function`, and `param_to_multi`.
- `device.rss`: first integrated source-shaped `tinygrad/device.py` metadata and allocator slice,
  covering canonical device strings, `BufferSpec`, `Buffer`/`MultiBuffer` descriptors, nbytes,
  allocation-state projection, refcounts, view offsets, explicit RSS byte-list allocation handles,
  copyin/copyout, deallocate, cache reuse, and cache freeing. The RSS struct is named `TGBuffer`
  to avoid a generated Rust backend collision, while the helper surface remains `buffer_*`. It now
  also exposes source-shaped `Device`/`Buffer`/`Allocator`/`Compiler`/`Compiled` facade names:
  canonicalization/default selection, available-device enumeration, buffer `size`/`base`/`ref`/
  `is_allocated`/`is_initialized`/`get_buf`/`ensure_allocated`/`allocate`/`deallocate`/
  `as_memoryview`/`copyin`/`copyout`, allocator `alloc`/`free`/`free_cache` and low-level
  `_alloc`/`_free`/`_copyin`/`_copyout`/`_map`/`_unmap`/`_encode_decode`, byte-list compiler
  `compile`/`compile_cached`/`disassemble`, renderer/compiler selection, device count,
  synchronize/finalize/profile hooks, and `enumerate_devices_str`. Native opaque runtime handles,
  real target imports, actual compiler cache persistence, and target-specific allocator backends
  remain later fidelity work.
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
  values now record total, known-bound, and runtime-var counts. The source-shaped public wrapper
  names from `engine/realize.py` are now present: `get_call_arg_uops`, `get_call_outs_ins`,
  `get_call_name`, `estimate_uop`, `resolve_params`, `unwrap_multi`, `exec_view`, `exec_copy`,
  `exec_kernel`, `exec_encdec`, `exec_validate`, `exec_graph`, `_validate`, `compile_linear`,
  `run_linear`, `time_call`, `track_stats`, `optimize_local_size`, `get_runtime`, and
  `get_graph_runtime`. These wrappers delegate to real implemented behavior where available
  (call metadata, estimates, parameter resolution, COPY/SLICE byte execution, PROGRAM staging,
  hosted PROGRAM execution through `run_linear`, per-context runtime-cache handles, and a
  deterministic static `optimize_local_size` rewrite for integer PROGRAM launch dimensions with no
  existing local size); profiling stats, timing-backed local-size benchmarking/cache, validation
  execution, and graph runtime remain explicit metadata/no-op placeholders. Dynamic symbolic
  launch dimension evaluation, general compiled numeric kernel invocation, validation execution,
  graph execution, full multi-buffer/device remapping, and dynamic-library runtime plumbing remain
  unported.
- `engine/jit.rss`: source-shaped `tinygrad/engine/jit.py` facade and metadata slice covering
  `prune_linear`, `create_graph_call`, `graph_split_rewrite`, `_copy_input`, `jit_lower`,
  `_check_no_non_tensor_return`, `graph_class`, `_buf_key`, `access_resources`,
  `updated_vars`, `updated_launch_dims`, `_access_resources`, `_all_devs`, `supports_uop`,
  `_written_uops`, `free_intermediates`, `_prepare_jit_inputs`, `add_linear`, and `reset`.
  The implementation builds real integrated UOps for graph packing (`CALL(CUSTOM_FUNCTION("graph"))`
  over a nested `LINEAR`), threads through the current `compile_linear`/memory planner/lowering
  path, records read/write resource metadata from call outs/ins, and exposes RSS state structs for
  graph runner, captured JIT, and TinyJit-style counters. Backend-specific graph launch, Python
  object identity/weakref behavior, exact tensor-return validation, input-copy aliasing rules, and
  full dynamic launch-dimension updates remain later fidelity work.
- `gradient.rss`: first integrated reverse-mode autodiff slice over interned UOp ids, covering
  target-specific symbolic gradients for `CAST`, `ADD`, `SUB`, `MUL`, `FDIV`, unary
  `NEG`/`RECIPROCAL`/`SQRT`/`EXP2`/`LOG2`/`SIN`/`TRUNC`, binary `POW`, `MAX` with upstream
  tie-splitting, mask-selection `WHERE`, zero gradients for integrated comparison predicates, and
  movement/reduce VJPs for `RESHAPE`, `EXPAND`, `PERMUTE`, `FLIP`, `PAD`, `SHRINK`, and `REDUCE`,
  plus grouped boundary VJPs for `CONTIGUOUS`, `CONTIGUOUS_BACKWARD`, `COPY`, `DETACH`, and `BITCAST`, then
  simplifying the resulting gradient graph. Source-shaped upstream entry points now cover
  `reduce_gradient`, `_compact_params`, `call_gradient`, `_deepwalk`, and `compute_gradient`
  over the current explicit-cache UOp model.
- `callify.rss`: first source-shaped `tinygrad/callify.py` graph-transform facade over interned
  UOp ids. It exposes the upstream transform names for tagging UOps, recognizing disk/creation
  copies, AFTER base projection, contiguous-to-store-after lowering, store-after-to-contiguous
  fallback, contiguous movement view construction, precompiled-call shells, final AFTER assignment
  collection, input buffer parameter replacement, and `transform_to_call`. The implementation
  records RSS `AllocCtx`/`CallifyResult` mappings and builds a CALL over a SINK body for supported
  graphs; full PatternMatcher parity, tag tuple semantics, exact precompiled-output rewriting, and
  schedule-cache normalization remain later fidelity work.
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
  Source-shaped symbolic entry points now cover `simplify_pow`, `const_arg`, `fold_const_alu`,
  `fold_add_divmod_recombine`, `lt_folding`, `canonicalize_simplex`, `fold_bitcast`,
  `gep_through_wmma`, `parse_valid`, `uop_given_valid`, `_valid_priority`, `simplify_valid`,
  `reduce_mul_chain`, `drop_and_clauses`, `where_on_load`, and `gated_given_valid` over the current
  interned UOp-id representation. `uop_given_valid` now has the upstream-shaped bounded fake-variable
  substitution path, multi-substitution support, closer valid-priority ordering, and the INDEX guard
  in `simplify_valid`, which lets masked index cleanup simplify bounded expressions such as
  `x % 4 -> x` under `x < 4`. Full byte reinterpretation for float bitcasts, exact WMMA/GEP
  layout pushing, range-clause dropping, and data-dependent load gating remain conservative/partial.
  The evaluator now covers the integrated unary math UOps
  `EXP2`, `LOG2`, `SIN`, `SQRT`, `RECIPROCAL`, `NEG`, `TRUNC`, plus binary `POW`, and handles
  scalar-to-tensor `EXPAND` materialization, `FLIP`, `PAD`, graph-backed `SHRINK` slicing, and
  multi-axis `REDUCE` materialization.
- `tensor.rss`: tensor surface over UOp ids with graph-building creation helpers (`full`,
  `zeros`, `ones`, `full_like`, `zeros_like`, `ones_like`, `arange`,
  `arange(start, stop, step)`, `linspace`, `eye`, bit-exact explicit-seed `rand`,
  `rand_like`, `uniform`, `randint`, `randperm`, `randn`, `randn_like`, `normal`,
  `normal_like`, `scaled_uniform`, `glorot_uniform`, `kaiming_uniform`, and
  `kaiming_normal`, plus checked argument-validation wrappers for `rand`, `uniform`,
  `randint`, `randn`, and `normal`), upstream-shaped scalar/wrapper helpers (`_uop`, `_wrap_uop`,
  `const`, `const_like`, `unique_const`, `_broadcasted`, and `_binop`),
  Python-facing public wrapper aliases for reverse/truediv/mod/fmod arithmetic,
  bitwise/shift helpers, `clamp`, `copysign`, `masked_fill`, `detach`,
  `contiguous_backward`, reduction/statistics, sort/topk, softmax/logsumexp, and
  cumsum/cumprod entry points,
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
  bitwise/logical `AND`/`OR`/`XOR`/`NOT`, integer shifts, floor/truncating div and modulo variants,
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
  plus upstream-shaped 2D padding normalization wrappers for scalar/list-like padding forms
  on pooling, convolution, transposed convolution, and max-unpool entry points,
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
  helpers, explicit result wrappers for `__bool__`, scalar-shape `__len__`, and `__delitem__`
  rejection, plus legacy false/unsupported sentinel helpers, and a metadata-state wrapper
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
  `data`/`item`/`tolist` helpers for supported CPU/value graphs, including zero-size `data`
  materialization and upstream-shaped singleton validation for `item`,
  graph-level `call`, `callify`, `linear_with_vars`, and `schedule_linear` wrappers over the
  existing `FUNCTION`/`CALL` and scheduler primitives, simple `assign` as STORE+AFTER with
  broadcast-to-target-shape, `clone` preserving param/grad state, `shard`/`shard_`/`shard_like`
  over the integrated multi-device UOp helpers including upstream `axis=None` replicated
  tuple-device COPY, and in-place binary/matmul wrappers composed as `op` plus `assign`,
  an explicit RSS-side `TensorRngState`/counter wrapper for `manual_seed` and `_next_counter`,
  plus state-returning random distribution wrappers for `rand`, `uniform`, `randint`, `randn`,
  and `normal` that thread the advanced counter state explicitly,
  multi-device `rand_like` construction through `MSTACK`/`MULTI`, and materialized 1D/2D
  `multinomial` sampling for supported CPU/value tensors,
  Tensor boundary helpers for invalid-sentinel creation (`invalids` over `uop_invalid`), shape/dtype/device
  `repr`, scalar-shape `len` rejection via sentinel return, bool rejection flag, contiguous-buffer graph
  boundary, and list-backed `numpy`/host-data materialization over the current evaluator,
  materialized `keccak` for row-wise `sha3_224`, `sha3_256`, and `shake_128` byte tensors plus
  a materialized `_hash_1mb` SHAKE-128 chunk/reduce helper over the current evaluator byte path
  with an upstream-shaped checked result wrapper for dtype/rank/1MiB row-size assertions,
  materialized byte/file boundary helpers for current local execution (`fs_load`/`fs_store`
  through `File.read_bytes`/`File.write_bytes`, `from_blob` as an external-pointer-shaped empty
  buffer placeholder, `from_url` through sync `Http.get`/`HttpResponse.bytes` with gzip byte
  decompression through `Gzip.decompress_bytes`,
  and `decode_hevc_frame` as an `encdec` `CUSTOM_FUNCTION` call dependency graph wrapper),
  explicit-size graph-shaped `masked_select`, evaluator-backed default-size `masked_select`,
  constrained 1D graph-shaped `nonzero`, and evaluator-backed N-D/default-size `nonzero`
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
  movement base recovery, movement argument extraction, and multi-node substitution.
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
  through the current memory planner. It also validates the `CALL(LINEAR, arg)`
  root path: slot-0 `PARAM` is resolved to the outer call argument and the held concrete buffer is
  not rewritten by memory planning.
- validates the first scheduler-to-realize bridge: supported scheduled `CALL(LINEAR, ...)` roots
  resolve inner `PARAM` args, run through `realize_run_linear`, and execute COPY, SLICE/view, and
  hosted PROGRAM calls with expected `TGBuffer` bytes. It also validates PROGRAM `var_vals`
  accounting for direct host execution and scheduled PROGRAM execution with a `BIND`.
- validates the first source-shaped scheduler memory planner slice: compute/copy lane separation,
  int8 arena `SLICE` rewrites that pass the integrated spec verifier, held-buffer exclusion, and
  same-lane slice-offset reuse after a buffer lifetime expires.
- validates the source-shaped scheduler indexing slice: upstream facade names are callable,
  `ALWAYS_CONTIGUOUS` classification, unconditional realize ops, non-contiguous copy-source
  realization, direct copy-store cleanup, and self-store source realization.
- validates scheduler indexing range/movement helpers: size-1 range folding, sequential range ids,
  `SHRINK`/`PERMUTE`/`FLIP`/`EXPAND` index mapping, validity-guarded `PAD` index mapping,
  and row-major `RESHAPE` div/mod index mapping plus identity simplification.
- validates scheduler indexing rangeify analysis records for realized roots, inherited movement
  chains, `PERMUTE` input-range swizzling, same-index multi-consumer range merging without
  unnecessary staging, EXPAND-triggered ending-range realization, and `REDUCE` `AXIS_REDUCE`
  injection.
- validates scheduler indexing rangeify graph rewrite and source-shaped facade wrappers for
  movement removal through `INDEX`, `REDUCE` metadata-to-range-source conversion, source `INDEX`
  insertion, PAD-to-`WHERE` locality wrapping, and
  device-carrying full-global `STAGE` option propagation.
- validates scheduler indexing rangeify `STAGE` option selection for full global staging,
  partial local staging, and upstream removable rules for always-contiguous vs non-contiguous children.
- validates the first scheduler rangeify facade slice by calling `get_kernel_graph`,
  `get_contiguous`, `_mop_index`, `flatten_bufferize`, `debuf`, `bufferize_to_store`,
  `remove_bufferize`, `remove_noop_bufferize`, and `split_store` against the integrated rangeify smoke graph,
  including global stage-to-store, local stage-to-`DEFINE_LOCAL` plus `BARRIER`, `AFTER`
  buffer reuse, removable stage range substitution, SHAPED_WMMA-to-WMMA/register lowering,
  CALL argument exposure, nested `INDEX` flattening, and elementwise `INDEX` pushdown.
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
  bindings, out-of-range bind rejection, bounded valid simplification of modulo indexes, and
  `simplify_valid_load` using that stronger symbolic path.
- validates the integrated source-shaped UOp renderer for precedence-aware symbolic expressions,
  `max`, inference `floordiv`, inference `BITCAST`, compact repeated `STACK`, UOp line output,
  first `pyrender` reconstruction for an ALU expression and bitcast, plus source-shaped render
  aliases for `pretty_print`, `print_uops`, `strip_binary_parens`, `srcs`, `render_marg`,
  `_render_with_splits`, and `pyrender`.
- validates UOp `ParamArg` metadata and definitions: `PARAM` slot/name/addrspace, bounded param
  vmin/vmax, `DEFINE_LOCAL`/`DEFINE_REG` addrspace, and rejection of invalid param bounds or
  negative local slots.
- validates upstream-shaped UOp device/buffer graph metadata: `DEVICE` tuple key/count,
  `UNIQUE`-backed `BUFFER` shape/device count, `COPY`, `MULTI`, `MSELECT`, `MSTACK`, `ALLREDUCE`,
  `CONTIGUOUS`, `DETACH`, `CUSTOM_FUNCTION`, and rejection of empty devices, out-of-range
  `MSELECT`, and invalid `ALLREDUCE` ops.
- validates source-shaped multi-device method helpers for `copy_to_device`, `multi`,
  `mselect`, `mstack`, `allreduce`, `detach`, and `contiguous_backward`, plus replicated
  `shard(axis=None)` tuple COPY and full-dtype preservation through `COPY`, `CONTIGUOUS`, and
  `DETACH`.
- validates the first source-shaped scheduler allreduce slice: naive `ALLREDUCE` over both
  `MULTI` and `MSTACK` inputs emits verifier-accepted `CONTIGUOUS -> MSELECT -> COPY` shard
  graphs reduced with `ADD`, plus explicit all2all/ring chunked graph builders.
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
  `!b.where(2,3) -> b.where(3,2)`. It also validates source-shaped symbolic helper
  aliases for `const_arg`, `fold_const_alu`, and `simplify_pow`.
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
  `1/(x*x) -> (1/x)*(1/x)`, weakint `(x+y)*c -> x*c+y*c`,
  `lt_folding` by gcd-constrained terms, simplex canonicalization for positive integer
  coefficients, and `(x%4)+(x//4)*4 -> x` div/mod recombination.
- validates simple symbolic `WHERE` folding for true, false, same-branch, and bottom-up
  constant-comparison conditions.
- validates integrated div/mod symbolic rewrites against upstream-shaped samples:
  `(n%8)//4 -> (n//4)%2`, `(n%8)%4 -> n%4`, `(b*6+2)//3 -> b*2`,
  `(b*6+2)%3 -> 2`, `(n*4+8)//6 -> ((n*2+1)//3)+1`,
  `(n*4+8)%6 -> ((n*2+1)%3)*2`, `(n*6+m)//6 -> m//6+n`,
  `(n%4+m)%2 -> (n+m)%2`, and `(n*4+m)//8 -> ((n*4+m)//4)//2`-style
  nest-by-factor lowering.
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
- validates source-shaped C-style renderer facade names over the same graph: renderer state
  construction, dtype/cast rendering, `_render`/`render`/`render_kernel`, vector typedef prefixes,
  supported dtype lists, aux parameter metadata, non-native float pattern hooks, bf16 cast node
  construction, WMMA metadata collection, fp8 index selection, OCML call rendering, CDNA arch
  predicates, and the HIP assembly facade.
- validates expanded function/mixin source wrappers through the active smoke: creation, dtype,
  elementwise, movement, RNG, and reduction mixin wrappers now call into the integrated tensor
  layer and are checked for representative dtypes, graph ops, shapes, and evaluated tensor sizes.
- validates expanded NN layer wrappers through the active smoke: Linear, Conv2d,
  ConvTranspose2d, BatchNorm, LayerNorm/LayerNorm2d, GroupNorm, InstanceNorm, RMSNorm, and
  Embedding constructors/calls execute over small integrated tensor graphs and are checked by
  evaluated output sizes.
- validates source-shaped LLVM/PTX/WGSL/NIR renderer backend facade names through the active smoke:
  LLVM dtype/constant/cast/WMMA/function/footer helpers, PTX value/memory/WMMA/modifier helpers,
  WGSL packed mask/load/type/nan helpers, and NIR symbol/type/ALU/cast/channel/immediate/param/
  vector/postrender helpers.
- validates source-shaped CPU/null/disk/Python runtime backend facade names through the active
  smoke and coverage audit: CPU worker/queue/signal/buffer helpers, null transfer/offset surface,
  disk open/close/io_uring/sharded-copy/absolute-offset surface including page-aligned shard
  metadata, and the Python `generic_wmma_helper` path over the existing `mk_wmma` graph API.
  These are deterministic RSS runtime state facades, not real mmap/io_uring/thread/native-kernel
  execution yet.
- validates first source-shaped target runtime facade batch through the active smoke and coverage
  audit: CUDA argument/time/system synchronization helpers, OpenCL status checks, Metal/WebGPU
  buffer/command helpers, AMD/NV/QCOM packet/register/allocation/profiling helpers, DSP RPC/lib
  helpers, RDMA work-queue encoding, and tinyfs connection/copy shells are present. These are
  deterministic RSS metadata/state facades; real target driver calls, native queues, memory maps,
  command submission, and profiling streams remain future work.
- validates first source-shaped runtime support and HCQ graph facade batch through the active
  smoke and coverage audit: file/MMIO helper names, HCQ queue/symbol/bind/signal/device hooks,
  TLSF/page-table/mapping allocator names, C ABI metadata helpers, CPU compiler object/JIT shells,
  ELF loader/relocation shells, and graph dependency/timestamp helpers are present. These are
  deterministic RSS state facades; native ioctl/mmap/eventfd/ELF relocation/clang/LLVM execution
  remains future work.
- validates focused runtime backend wrapper batch through the active smoke and coverage audit:
  Metal graph launch/timestamp/support predicates, HIP device/program allocation and transfer
  shells, NPY allocation/transfer shells, and QCOM compile/check/read/disassemble wrappers are
  present. These remain deterministic RSS state facades, not real native backend execution.
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
- validates the expander slice: source-shaped helper names are callable, scalar `CONTRACT`
  lowers to `STACK`, `CONTRACT` of
  `UNROLL` lowers to indexed `GEP`, empty `UNROLL` unwraps, double `UNROLL` merges axes,
  `END` consumes `UNROLL` axes through `CONTRACT`, same-axis elementwise `UNROLL` sources are
  expanded, mixed-axis `UNROLL` sources are swizzled through `GEP`, scalar non-unroll sources
  are broadcast, and `GEP` args expand across lanes. It also validates the first
  `pm_pre_expander` rules for unrolled ranges plus REDUCE/STORE unroll contraction, and
  validates `GROUP_REDUCE` lowering through local `STAGE` plus final `AXIS_REDUCE` loops.
- validates the devectorizer slice: upstream facade names are callable, vector `ADD`, vector `CAST`, and vector `BITCAST`
  scalarize into `STACK` nodes of scalar lane operations and pass the integrated UOp spec
  verifier. It also validates the first `pm_render` normalizations for vector `CONST`,
  multi-lane `GEP`, scalar `GEP(0)`, and single-source `STACK`, plus the first context-free
  load/store folding rules for stack-backed `INDEX`, `GEP`, `PTRCAT`, and vector load/store splitting.
- validates the first `do_linearize` wrapper by appending a cleaned `LINEAR` child to a
  `PROGRAM(SINK, DEVICE)` and passing the integrated UOp spec verifier.
- validates the first `do_estimates` wrapper by adding `PROGRAM` estimate metadata for the sample
  linear stream: `ops=128`, `lds=512`, and capped unique memory `mem=64`.
- validates the first `do_render`-style wrapper by rendering a cleaned `LINEAR` stream into a
  non-empty `SOURCE` child on `PROGRAM(SINK, DEVICE, LINEAR)`, passing the integrated UOp spec
  verifier, and producing parseable C for the sample gated-store kernel.
- validates the `codegen.gpudims` missing-local global-store guard: `add_gpudims` creates a
  `WHERE(..., idx, INVALID)` index when a local range is absent from a global store index, and
  `gater_move_gate_from_index` lowers that shape to a gated `STORE`. It also validates the CStyle
  path applies the gpudims/gater pre-linearization orchestration: the resulting `LINEAR` has
  `IF`/`ENDIF`, no gated `STORE`, and renders to parseable C.
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
- validates explicit RSS RNG state threading through a chained smoke path: `manual_seed(42)`,
  state-returning `rand`, `uniform`, `randint`, `randn`, and `normal`, with the final counter
  state surfaced to callers.
- validates upstream-shaped random distribution argument checks through result wrappers:
  `rand` rejects non-float dtype and negative shape, `uniform` rejects `low >= high`,
  `randint` rejects non-int dtype and `low >= high`, `normal` rejects negative `std`, and
  valid `uniform` returns `ok`.
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
- validates upstream-shaped 2D padding normalization wrappers through the active tensor smoke:
  `(h,w)` padding for avg-pool and conv2d, `(left,right,top,bottom)` padding for max-pool
  values/indices, and `(h,w)` padding/output-padding for transposed convolution.
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
- `nn/*`: first source-shaped facade batch for `tinygrad/nn/__init__.py`, `nn/optim.py`,
  `nn/state.py`, and `nn/datasets.py`. `calc_stats`, 1D convolution layer constructors,
  RMS-style `_norm`, embedding forward/backward materialized helpers, optimizer entry points
  (`SGD`, `Muon`, `AdamW`, `Adam`) over explicit `TensorState` lists, state-dict/list helpers,
  tensor byte-IO shells, archive/load/save shells, deterministic `mnist`/`cifar` dataset split
  facades, simple safetensors metadata parsing/loading, uncompressed tar regular-file extraction,
  and the first ONNX runner tensor execution slice are present. The active smoke calls these entry
  points and checks constructor, optimizer/state list, IO, embedding, dataset shape, state-file,
  tar extraction, and ONNX tensor-op behavior. This does not yet implement exact upstream module
  objects, Python optimizer mutation semantics, complete safetensors/torch/zip/compressed-tar
  formats, real dataset downloads/parsing, complete ONNX parsed-model execution, or enough backend
  behavior to train real examples.
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
- validates public Tensor wrapper aliases through the active tensor smoke: reverse/truediv
  arithmetic, `rpow`, Python-style and C-style mod aliases, bitwise/shift/not aliases, `clamp`,
  `copysign`, `masked_fill`, `detach`, `contiguous_backward`, all/axis arg reductions, sort,
  argsort, topk, all/axis statistics aliases, softmax/log_softmax/logsumexp aliases, and
  cumsum/cumprod aliases.
- validates Tensor dynamic indexing helpers through the active tensor smoke: default-size
  `masked_select`, default-size 2D `nonzero`, and padded/truncated explicit-size 2D `nonzero`
  on materialized/evaluator-backed tensors.
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
- validates materialized hash boundary helpers: row-wise `keccak`, permissive evaluator-backed
  `_hash_1mb` for local smoke data, and the upstream-shaped checked `_hash_1mb` wrapper rejecting
  non-uint8 tensors and non-1MiB row widths.
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
- validates Tensor object-magic result wrappers through the active smoke: `len` succeeds on
  non-scalar tensors, `bool` is rejected with a TypeError-shaped message, and `delitem` is
  rejected with a TypeError-shaped message.
- validates representative unary/binary math Tensor ops against real tinygrad: `neg`,
  `reciprocal`, `sqrt`, `rsqrt`, `log2`, `log10`, `exp2`, zero-input `sin`, `trunc`, and `pow` on exactly
  representable or precision-stable values.
- validates composed Tensor elementwise helpers against real tinygrad: `sign`, `abs`, `square`,
  `minimum`, `clip`, `ceil`, `floor`, `sigmoid`, `exp`, `log`, `cos`, `tan`, `tanh`,
  `round`, `sinh`, `cosh`, `atanh`, `asin`, `acos`, `atan`, `asinh`, `acosh`, `erf`,
  `leaky_relu`, `quick_gelu`, default tanh-approx `gelu`, `swish`/`silu`, `hardswish`,
  `hardsigmoid`, `softplus`, `mish`, `logsigmoid`, `elu`, `celu`, `selu`, `softsign`, `lerp`,
  `hardtanh`, `relu6`, and upstream-style `relu`, including positive, negative, and zero values.
- validates bool and int bitwise/logical Tensor ops against real tinygrad: `AND`, `OR`, `XOR`,
  and signed/unsigned/bool `bitwise_not` aliases.
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
  partially integrated now. Source-shaped mixin/function facade names are now present for dtype
  aliases, reverse bitwise/mod/shift operators, movement/view helpers, random-bit helpers, `einsum`,
  and Function context saves, but exact upstream class-global RNG mutation/counter identity
  beyond the explicit RSS state-threaded wrappers,
  Python exception behavior for `__bool__`/`__len__`/`__delitem__`, true ndarray/buffer object parity,
  full Python `Tensor.__init__`/object identity behavior,
  weakref/all-tensor registry updates, Python-object `backward` grad mutation, full view-assign substitution and realized-buffer mutation safety checks, exact Python-global
  seed/counter APIs, broader distribution helpers,
  full lazy/batched Householder QR and full-matrices/upstream Jacobi SVD parity,
  exact lazy hash graph semantics beyond the current materialized `_hash_1mb` checked path,
  exact tinyfs chunk-tree storage semantics for `fs_load`/`fs_store`, real external-pointer buffer
  ownership/lifetime for `from_blob`, exact cache/progress behavior for `from_url`, and
  actual HEVC decode runtime execution behind the `encdec` custom function,
  real packed image convolution kernels and lazy Winograd convolution integration rather than the
  current direct-conv fallback,
  full symbolic/lazy indexing semantics beyond the current evaluator-backed default-size
  `masked_select`/`nonzero` paths,
  exact lazy `einsum` contraction parsing, broader composed math coverage,
  full conv/pool semantics including more negative/asymmetric edge cases and broader dimensionality,
  remaining Python-facing method breadth such as exact object magic and
  Python-level overload/exception parity,
  and exact tinygrad semantics are still incomplete.
- `function.py` and `gradient.py`: autograd is partially integrated for scalar symbolic UOp
  graphs, simple Tensor movement/reduce graphs including slice/pad, mask selection, max, and
  selected unary/math VJPs. Source-shaped gradient entry points are present for reduce gradients,
  compacting params, call gradients, deep-walk path discovery, and compute-gradient maps.
  `tensor_gradient` now wraps the supported implicit scalar-gradient and explicit incoming-gradient
  paths for a list of target UOps, `backward` forwards to those explicit-target paths, and RSS
  `TensorState` helpers model `.grad` discovery/set/clear/accumulation for explicit state lists,
  but full Function-style APIs, complete VJP coverage, Python weakref-driven Tensor discovery and
  in-place Python object mutation, and exact tinygrad gradient semantics are still missing.
- `callify.py`: source-shaped entry points are present and build CALL metadata for supported SINK
  graphs, but exact PatternMatcher rewrite parity, Python tag tuple semantics, precompiled-output
  redirection, and cache-key buffer replacement semantics are still incomplete.
- `uop/ops.py`, `uop/upat.py`, `uop/symbolic.py`, `uop/divandmod.py`, `uop/decompositions.py`, `uop/render.py`, `uop/spec.py`, `uop/validate.py`: full
  source-aligned implementation. UPat/rewrite and a broader method-helper/alias slice are integrated now, but
  the full upstream method surface, symbolic/decomposition/divmod coverage, exact symbolic shape
  representation, real upstream `Buffer`/`MultiBuffer` realization semantics, complete render/pyrender
  behavior, full spec validation, full Z3-backed bounds validation, and exact upstream semantics are still incomplete.
- `schedule/*`: source-shaped scheduler remains partial. The first dependency-ordering slice in
  `schedule/__init__.rss` and first lifetime-reusing memory planner slice in `schedule/memory.rss`
  are integrated, and `schedule/indexing.rss` has realize-map generation, first range/movement
  helpers including PAD and RESHAPE, first RESHAPE identity/per-coordinate symbolic
  simplification, first rangeify analysis records including same-index multi-consumer merging,
  first EXPAND-origin ending-range realization,
  first device-aware full-global staging options, and first rangeify graph
  rewrite to `STAGE`/`INDEX`. Exact TLSF split/coalesce semantics, full upstream sink-level RESHAPE simplification,
  multi-kernel behavior, schedule caching, linear-call resolution, variable binding extraction,
  and JIT capture plumbing remain.
- `codegen/*`: full lowerer and late passes aligned to tinygrad. The first linearizer priority
  toposort, CFG/control-flow insertion, split-end cleanup, gated-store line cleanup,
  gate-from-index movement, first expander CONTRACT/UNROLL/END normalization, same/mixed-axis expansion, and `pm_pre_expander` REDUCE/STORE/range unroll handling,
  `PROGRAM` -> `PROGRAM+LINEAR` wrapper, integer upper-bound `do_estimates` metadata,
  first `ProgramInfo.from_sink` metadata derivation including integer `SPECIAL` launch dimensions, and
  `PROGRAM+LINEAR` -> `PROGRAM+LINEAR+SOURCE` CStyle wrapper are integrated, but pre-existing
  IF rejection, remaining late expansion/devectorization, range simplification, broader upstream
  rewrite-pipeline parity, ISA/regalloc, full symbolic estimates, compile/binary, and clear scope
  for GPU-only optimization passes remain.
- `renderer/*`: exact target-specific compiler integration, full prefix emission, real LLVM/PTX/
  WGSL/NIR emission, Mesa/LLVM/CUDA native bindings, exact AMD instruction tables, x86 lowering,
  real machine-code assembly/disassembly, and renderer policy beyond the source-shaped facade
  names now in `renderer/*.rss`.
- target-specific `runtime/*`, `runtime/support/*`, native device backends, and full JIT graph
  execution: source-shaped CPU/null/disk/Python facades, first target runtime facades, and
  HCQ/support graph facades are present, but real mmap/io_uring/thread/native-kernel execution,
  native CUDA/Metal/WebGPU/OpenCL/AMD/NV/QCOM/DSP/RDMA driver calls, native clang/LLVM/ELF
  relocation, support-layer HCQ semantics, and device/runtime integration still need to be ported.
  The generated `runtime/autogen` data has been copied, but handwritten behavior must still be
  ported and accounted for.
- `nn/*`: deeper layer/module semantics, optimizer algorithms, model state serialization, and real
  dataset loaders. The first source-shaped facade names are present, but exact upstream behavior is
  still incomplete. `nn/onnx.py` now has a first real tensor execution slice, but complete parsed
  ONNX model parity, opset handling, control-flow ops, quantization, and broad operator coverage
  remain separate follow-up work.
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
