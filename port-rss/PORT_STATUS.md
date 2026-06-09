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
| `dtype.py` | scalar `DType`+`itemsize`; `dtypes` namespace (exact values, lines 176-197); `is_float/is_int/is_unsigned/is_bool`; **`can_lossless_cast`** (lines 255-272) | ✅ `dtype.rss`, all cross-checked equal to tinygrad |

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
- No `;`-multi-statement lines; parenthesized sub-expressions in arithmetic (`(a+7)/8`,
  `o + (a*b)`) rejected — must split into `let`s.
- No struct-field shadowing across sibling blocks (function-wide unique locals, by design).
- These are exactly the frictions to fix to make a real tinygrad port tractable in rss.

## rss/mc bugs found and FIXED during the port
- **rss parser**: parenthesized sub-expressions in arithmetic (`(a+b)`, `(a+b)*c`) were rejected (RS0015) due to a `trim_outer` double-call shadowing bug in `parse_expr`. Fixed; regression fixture added; `cargo test` green. (rsscript commit f49ac9a)
- (earlier) mc: negative float literals, float array-element arithmetic, exp/log/tanh intrinsics; rss: Hashable/Eq.
