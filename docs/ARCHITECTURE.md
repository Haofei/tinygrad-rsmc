# tinygrad-rsmc — architecture & plan

Port of tinygrad with the pipeline split across two self-made languages:

- **rsscript (rss)** — the frontend: `Tensor → UOp graph → rewrite/schedule → codegen
  lowering → render kernel source`. Pure graph + string work; no in-process FFI.
- **modern-c (mc)** — the kernel backend: rendered MC kernels + a static host harness,
  compiled `mcc emit-c → clang` to a native binary.

Twin purpose: **harden rss + mc** by driving real features out of a real workload, and
**learn tinygrad** internals in the process.

## Why this split works (the FFI wall dissolves)

tinygrad's CPU backend does three things (`tinygrad/runtime/ops_cpu.py`):
1. **render** kernel source (pure string work) — `renderer/cstyle.py`
2. **compile** via subprocess / compiler lib (pure-ish)
3. **load & call**: `mmap` RWX + `ctypes` function pointer ← the only part that needs FFI

In this port, **step 3 never happens inside rss**. rss is a *source-to-source compiler
driver*: it emits a self-contained MC program (kernels + statically-sized harness), shells
out (`mcc → clang`) via the Process API, runs the binary, and reads results back through a
file. No `dlopen`, no in-process function pointers.

Because tinygrad knows every shape at schedule time, the generated MC harness has
**statically-sized buffers** — so MC needs no dynamic collections. MC's checked-C / trap
story is a free safety bonus on the kernels.

`tinygrad/runtime/ops_python.py` (a pure-Python UOp interpreter, no compiler/FFI) is
precedent that a no-FFI execution path exists.

## Language division

| Layer | Lang | tinygrad source |
|---|---|---|
| Tensor API + autograd | rss | `tensor.py`, `function.py`, `gradient.py` |
| UOp graph + rewrite engine (the core) | rss | `uop/ops.py` (`UOp`, `Ops`, `PatternMatcher`/`UPat`) |
| dtypes | rss | `dtype.py` |
| scheduler (graph→kernels) | rss | `schedule/` |
| lowerer + optimizer (UOp→linear kernel) | rss | `codegen/` |
| MC kernel renderer (new) | rss emits MC | analog of `renderer/cstyle.py` |
| compiled kernels + static host harness | mc | generated |
| build/run driver (mcc+clang, exec, read) | rss | analog of `device.py` `Compiler`/runtime |

Renderer extension points in tinygrad: a `Renderer` subclass mirroring
`ClangRenderer(CStyleLanguage)` (`cstyle.py:228`) and a `Compiler` subclass (`device.py:276`,
override `compile`).

## Required language features (drive the hardening)

rss (frontend):
1. **Hashable/Eq for user-type Map/Set keys** — unblocks UOp interning/dedup/memo. (task #1)
2. Memo/@cache helper (builds on #1). (task #3)
3. Operator overloading, pure/declared-effect only — Tensor/UOp ergonomics. (task #4)
4. Ergonomic typed enum payloads + exhaustive match for `UOp.arg`. (task #5)
5. varargs + default/keyword args — Tensor API ergonomics. (task #6)
6. (later) real FFI via reserved `ffi`/`device` markers → in-process JIT instead of subprocess.

mc (backend):
1. **Opt-in hosted target profile with explicit, fallible I/O** — data round-trip. (task #2)
2. Float math intrinsics (exp2/log2/sin/cos/sqrt) via libm extern — real kernels. (task #2)
3. (optional) f16/bf16 + float4 vector types — dtype/vectorization coverage.
4. (optional) dynamic Vec/HashMap (allocator-passed) — only if size-generic kernels wanted.

Principle guardrails (do NOT break):
- rss: keep the surface small/reviewable; features must be pure/total or go through an
  existing capability gate (the `ffi` marker). Avoid a dynamic `Any` — use typed enums.
- mc: keep kernel/freestanding profile the default; I/O is a SEPARATE opt-in profile,
  explicit + fallible (Result/trap), allocator passed in. No ambient libc.

## Phases

- **Phase 0** — spike the process+file boundary (covered by mc task #2's round-trip demo).
- **Phase 1** — build `MCRenderer` + an MC `Compiler` *inside tinygrad's Python*, validated
  against tinygrad's own tests. Produces reference MC kernels + a byte-for-byte oracle for
  the rss port. (no feature deps — in progress, see `oracle/`)
- **Phase 2** — port frontend to rss slice-by-slice: `dtype → uop/ops (PatternMatcher) →
  minimal Tensor → schedule → codegen lowerer → port the MC renderer`. Diff rendered MC vs
  the Phase-1 oracle.
- **Phase 3** — autograd (`function.py`/`gradient.py`) + a tiny nn; train a 2-layer MLP.

## Reference kernel shapes (the oracle)

Captured from tinygrad ClangRenderer (NOOPT=1) — see `oracle/c_reference/`.

Elementwise add (size 8):
```c
void E_8(float* restrict data0_8, float* restrict data1_8, float* restrict data2_8) {
  for (int Lidx0 = 0; Lidx0 < 8; Lidx0++) {
    float val0 = (*(data1_8+Lidx0));
    float val1 = (*(data2_8+Lidx0));
    *(data0_8+Lidx0) = (val0+val1);
  }
}
```

Reduce sum (size 8 → 1):
```c
void r_8(float* restrict data0_1, float* restrict data1_8) {
  float acc0[1];
  *(acc0+0) = 0.0f;
  for (int Ridx0 = 0; Ridx0 < 8; Ridx0++) {
    float val0 = (*(data1_8+Ridx0));
    *(acc0+0) = ((*(acc0+0))+val0);
  }
  *(data0_1+0) = (*(acc0+0));
}
```

## C → MC syntax mapping

| C (tinygrad) | MC | notes |
|---|---|---|
| `void NAME(...)` | `export fn NAME(...) -> void` | |
| `float* restrict dataN` | `dataN: *mut f32` (TBD: or `*mut [N]f32`) | **pointer/index form TBD** — adopt the convention mc task #2's hosted demo proves compiles |
| `float x = v;` | `let x: f32 = v;` (`var` if mutated) | |
| `float acc0[1];` | `var acc0: [1]f32;` | reduce accumulator |
| `for (int i=0;i<N;i++){…}` | `var i: usize = 0; while i < N { … i = i + 1; }` | **MC has no C-for; only `while`** → RANGE emits init+while, END emits incr+close |
| `*(ptr+idx)` | `ptr[idx]` or `raw.load<f32>(...)` | depends on chosen pointer form |
| `*(ptr+idx) = v;` | `ptr[idx] = v;` or `raw.store<f32>(...)` | |
| `(a+b)`, `(a*b)` | same | MC infix arithmetic matches |
| `0.0f` | `0.0` | MC float literal |
| `INFINITY`/`NAN` | TBD | from mc mathf surface |
| `sqrt/exp2/log2/sin` | mc libm extern intrinsics | from mc task #2 |

Open question resolved by mc task #2: the **buffer-passing ABI** (how the harness lays out
input/output buffers and passes them to a kernel) — the renderer's pointer/index form and
the harness generator must agree on it.

## Status

- [x] Confirmed toolchain (python3.14, clang, zig) and tinygrad runs on CPU/clang.
- [x] Captured oracle C kernel shapes (`oracle/c_reference/`).
- [x] Phase-1 `MCRenderer` (`oracle/mc_renderer.py`): elementwise, reduce, select, matmul
      (nested loops + reduce), 2D broadcast. Dual C/MC generator `oracle/gen_oracle.py`.
- [x] rss Hashable/Eq feature landed (subagent, task #1) — branch `feat/hashable-map-keys`.
- [x] mc hosted-I/O + mathf landed (subagent, task #2) — branch `feat/hosted-io-and-mathf`.
- [x] **All generated MC kernels compile** mcc -> C -> clang.
- [x] **Numerical round-trip PROVEN** (`oracle/roundtrip.py`): add, mul, affine, relu, sum
      all produce results identical to tinygrad's CPU backend, through the full
      tinygrad -> MC -> native pipeline. **Phase 1 complete.**
- [ ] Phase 2: port the frontend to rss, diffing rendered MC against this oracle.

### MC backend findings — validated against mcc (great hardening signal)

The locked kernel/buffer convention (all confirmed by compiling + running):
- Global buffers are `PAddr`; accessed via `raw.load/store<T>(pa_offset(buf, idx*itemsize))`
  inside an `unsafe {}` block. (`*mut f32` indexing is rejected: arrays/slices only.)
- Reduce accumulators are local fixed arrays (`var acc: [n]f32 = uninit;`) with `acc[i]`.
- reg-vs-global is decided by **addrspace** (REG vs GLOBAL), not src op (acc reads index
  through an `AFTER` node, so an op check misses them).
- Index arithmetic unified to `usize` — MC forbids implicit int<->usize conversion.

MC language gaps the port surfaced (candidate MC hardening features):
- **`if` is a statement, not an expression** — `WHERE`/select lowered to a generated pure
  `select_T(c,a,b)` helper. A MC if-/ternary-expression would remove the helper.
- **Negative float literals & `a + (-c)` are rejected** by the MIR verifier; only
  subtraction by a positive literal works. Renderer rewrites `ADD(x, neg_const)` ->
  `(x - |c|)`. General `NEG`/negative-literal support is the obvious MC follow-up.

### rss frontend finding (affects Phase 2 design)
- rss v0.6 can `match` sum-payload variants but **cannot construct** them (RS0206). `UOp.arg`
  is heterogeneous; the port must encode it without constructing payload variants (e.g.
  parallel typed fields, or extend rss — relates to task #5).

### Phase-1 findings (drive Phase-2 + feed the language tasks)

1. **Index/RANGE arithmetic must be one integer type.** Loop vars render `usize` but index
   ALU temps render `i32` (tinygrad uses C `int` freely). MC forbids implicit numeric
   conversion, so `data[(alu0_i32 + Ridx0_usize)]` would be a type error. Fix in the MC
   renderer: render all index math + buffer indices as `usize` (or insert explicit casts).
   This is the first real MC-strictness interaction the oracle caught.
2. **No C ternary in MC.** `WHERE` (select, used by relu/max) emits a placeholder MC
   if-expression `(if c { a } else { b })` — exact expression-conditional syntax to be
   confirmed against mcc (mc task #2).
3. **Float intrinsics + INF/NAN** (exp2/log2/sin/sqrt) still map to C builtins; need the
   mc mathf surface (mc task #2) before activations beyond relu render.
4. **Pointer/index form** currently `dataN[idx]` on `*mut f32`; must match the buffer ABI
   mc task #2 establishes (could become `raw.load<f32>` etc.).
5. Cosmetic: function-name suffixes drift (`E_8` vs `E_8n1`) due to a process-global kernel
   counter when rendering the same AST twice; not a correctness issue.
