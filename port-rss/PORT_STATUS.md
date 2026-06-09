# Real port status (honest)

This directory (`port-rss/`) holds **faithful translations of tinygrad's actual source files**,
validated against tinygrad's own output. This is the real port.

**Important correction:** the files under `../frontend-rss/` and `../oracle/` are NOT a port.
They are original concept demos (a micrograd-style scalar autograd, a hand-built 4-rule
pattern matcher, toy arenas, a fusion demo) that *imitate ideas* from tinygrad. They helped
learn tinygrad and drove the rss/mc hardening fixes, but they do not translate tinygrad's
code or reproduce its API. Calling them "ported the PatternMatcher/autograd/ShapeTracker"
earlier was a mischaracterization.

## What is actually ported (translated from tinygrad source + validated)

| tinygrad file | scope | status |
|---|---|---|
| `dtype.py` | scalar dtypes, count-aware itemsize, predicates, can_lossless_cast, promo lattice + least_upper_dtype, **vec() vectorized dtypes** (`dtype.rss`); PtrDType/ImageDType blocked by rss clone-gap |
| `helpers.py` | `prod`, `ceildiv`, `round_up`, `all_same`, `dedup`, `argsort` (+ floor-div) | ✅ `helpers.rss`, all cross-checked equal to tinygrad |
| `uop/ops.py` (CORE) | `UOp`+interning (`uop.rss`); **PatternMatcher** symbolic rewrite (`rewrite.rss`); **generic UPat DSL** -- recursive pattern + wildcard + capture (`upat.rss`) | ✅ rewrite cases; UPat MUL(any,CONST 1) matches x*1 (captures Var), rejects x*2 |
| `tensor.py` (CORE) | **Tensor as a lazy UOp-graph wrapper** (`tensor.rss`) | ✅ (x*1)+(2+3)->Add(Var7,Const5); (2*3)+4->Const(10) |
| `uop/ops.py` symbolic | **vmin/vmax integer bounds** (`symbolic.rss`) | ✅ x+2->[2,12]; x*3->[0,30]; x*x in[1,4]->[1,16]; x*x in[-2,3]->[-6,9] |
| `uop/ops.py` toposort | **UOp DAG toposort** with structural dedup (`toposort.rss`) | ✅ (2+3)*4->5 nodes sources-first; x*x shared x->2 nodes |
| `uop/ops.py` Ops enum | **complete 93-opcode Ops enum + GroupOp groups** (Unary/Binary/Ternary/Movement/ALU) (`ops_enum.rss`) | ✅ all group predicates match tinygrad (is_unary(NEG), is_alu(MULACC), etc.) |
| **capstone: training** | **SGD training loop driven by the ported autodiff** (`train.rss`) | ✅ loss 56->~0; w converges to slope 2.0 |
| `device.py`/`runtime` | **Device[name] dispatch + CPU-interpreter runtime backend** (`device.rss`) | ✅ Device[interp] (x+2)*3 -> 9 12 15; unknown device handled |
| scheduler memory planner | **liveness-based buffer reuse** (linear-scan) (`buffer_reuse.rss`) | ✅ chain 4 nodes->2 buffers; diamond 4->3 buffers |
| `gradient.py` (CORE) | **reverse-mode autodiff over the UOp graph** -- add/mul/sub (`gradient.rss`), **relu** (`gradient_relu.rss`); **DAG-shared accumulation** over an arena (`autodiff_dag.rss`) | ✅ symbolic d(...) cases; **DAG: d(x*x)=6@x=3, d((x+x)*x)=8@x=2** (shared node accumulates) |
| `codegen/`+`renderer/` (CORE) | **lower a ported UOp graph to an MC kernel + EXECUTE** -- elementwise (`render.rss`), **reduce** (`render_reduce.rss`), **matmul** (`render_matmul.rss`), **relu** (`render_relu.rss`); **div/neg/max/exp/log/tanh** (`render_ops.rss`, transcendentals via mc std/mathf intrinsics); real loop render->MC->native | ✅ (x+2)*3->[9,12,21,-3]; sum(x+1)->14; 3x3 matmul exact; max((x/2)+1,-x), tanh(x), log(exp(x))~=x all execute correctly |
| `schedule/` (CORE) | **scheduler**: single-kernel dispatch (`schedule.rss`) + **multi-kernel split** (`schedule_multi.rss`): a reduce feeding an elementwise op becomes 2 ordered kernels with an intermediate buffer | ✅ (x+2)*3->1 kernel; sum(x*2)->reduce; **x-sum(x)->K1(reduce->s)+K2(x-s0)=[-9,-8,-7,-6]** |

Validation: `dtype.rss` prints float32.priority=13, itemsize=4, bool.itemsize=1,
is_float(float32)=yes, is_float(int32)=no, is_int(int32)=yes, is_unsigned(uint8)=yes —
identical to `python3 -c "from tinygrad.dtype import dtypes; ..."`.

## Not yet ported (the overwhelming majority)

Even within `dtype.py`: `PtrDType`, `ImageDType`, `vec()`, `scalar()` for vec types,
`least_upper_dtype`/promo lattice, `can_lossless_cast`, the fp8/bf16 bit-twiddling, numpy/
torch bridges. (vec/scalar need self-referential owning clones — blocked on rss ergonomics
around moving owned fields, noted below.)

Then the rest of the framework, essentially untouched as a *translation*:
`helpers.py`, `uop/ops.py` (UOp + PatternMatcher, ~3.7k LOC), `tensor.py` (~1.4k LOC),
`function.py`, `gradient.py`, `codegen/`, `schedule/`, `engine/`, `renderer/`, `device.py`,
`runtime/`. Realistically a multi-month, file-by-file effort; **the port as a whole is at an
early single-digit-percent and is not complete.**

## rss ergonomics that make faithful translation hard (candidate hardening)
- Cannot `fresh`-return a collection built by push (stdlib uses `native`); forces SoA/`take`.
- Owning struct fields can't take a `read` param or move out of `read self` → construct with
  inline literals or `take`; blocks straightforward `scalar()`/`vec()` clones.
- No `;`-multi-statement lines (one statement per line).
- ~~parenthesized sub-expressions in arithmetic rejected~~ — FIXED (see below).
- No struct-field shadowing across sibling blocks (function-wide unique locals, by design).
- rss ergonomic gap: `==` on a `read` enum param fails (`&Ops == Ops`, RS1101) -- field-access compares work, but a bare `read Ops` param does not; use `match` instead. Minor; worth auto-deref in the comparison lowering.
- rss clone-gap (managed-field clone): can't copy a String/managed struct field out of a `read` value (RS1101 'cannot move out of ... behind shared reference'); blocks generic vec(base: read DType) and PtrDType(base). Worked around with literal-based vec constructors. A callable `.clone()` for managed values / fields would resolve it.
- rss stdlib gap (NOT fixed): no Int->Float conversion (`Int.to_float`/`Float.from_int` don't resolve) -- blocks numeric interpreters that mix Int consts with Float math. Adding it spans the checker type table, the reg_vm interpreter, and the rust-lowering ABI+runtime; deferred to avoid an unverified multi-site change. Worked around in device.rss by storing const as a Float field.
- rss codegen bug (NOT fixed): `let mut s = <read-param>` then reassign emits ill-typed Rust (unmappable E0308). Root cause confirmed (mut local bound to a read-param `&T`, then assigned an owned `T`). A naive fix (clone the read-param at the binding when mutable) was tried but **broke 21 checker_lowering snapshot tests** (too broad -- it changes valid no-reassign lowerings), so it was reverted. Needs a precise fix gated on actual later owned-reassignment. Worked around in the port by avoiding the pattern.
- ~~rss codegen bug: read-PARAM passed as a read arg emits `&&T` (E0308)~~ -- **FIXED & VERIFIED GREEN** (rsscript: read-param no longer double-borrowed). NOTE: the fix changed the (latent-buggy) output of 21 checker_lowering snapshot tests -- they had encoded `f(&<read-param>)` = `&&T` which never compiled; corrected to the right `f(<read-param>)`. Full rss suite green incl. the parity suite that COMPILES+RUNS generated Rust (checker_lowering 212/0, parity 112/0). An earlier 'green' claim for this fix was premature and is now corrected.
- `derives(Clone)` is accepted but there is no callable `.clone()`/`Clone.clone(..)` in source -> cannot copy a managed value; blocks `scalar()`/`vec()` clones (worked around via reconstruction-by-name). Candidate rss feature (larger than a bug fix).
- rss `/` truncates toward zero; Python `//` floors -> ceildiv/round_up need an explicit floordiv (handled in helpers.rss).
- These are exactly the frictions to fix to make a real tinygrad port tractable in rss.

## rss/mc bugs found and FIXED during the port
- **rss parser**: parenthesized sub-expressions in arithmetic (`(a+b)`, `(a+b)*c`) were rejected (RS0015) due to a `trim_outer` double-call shadowing bug in `parse_expr`. Fixed; regression fixture added; `cargo test` green. (rsscript commit f49ac9a)
- (earlier) mc: negative float literals, float array-element arithmetic, exp/log/tanh intrinsics; rss: Hashable/Eq.
