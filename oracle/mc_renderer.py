"""
Phase-1 MCRenderer: renders tinygrad's linearized UOp kernels to modern-c (MC) source.

Lives inside the tinygrad/Python world deliberately (Phase 1 of docs/ARCHITECTURE.md):
it reuses tinygrad's entire working frontend to emit MC kernels, so we get a *reference*
MC renderer + a byte-for-byte oracle BEFORE porting anything to rss.

Scope right now: f32 elementwise + reduce kernels (the oracle in oracle/c_reference/).
Known TODOs (parameterized, to align with mc task #2's proven buffer/IO convention):
  - pointer/index form: currently `dataN[idx]` on a `*mut f32` param. If mcc requires a
    different form (e.g. `raw.load<f32>`), only INDEX/LOAD/STORE + render_kernel change.
  - CAST / gated-LOAD / float4 / non-f32 dtypes: not needed for Phase-1 kernels.

Usage:  see gen_oracle.py
"""
from typing import cast
from collections import defaultdict
from tinygrad.uop.ops import Ops, UOp, GroupOp, PatternMatcher, UPat, range_str, axis_letters
from tinygrad.dtype import dtypes, DType, PtrDType, AddrSpace
from tinygrad.renderer.cstyle import CStyleLanguage, ClangRenderer, base_rewrite
from tinygrad.helpers import dedup as dedup_dtypes

# MC-specific per-UOp text. Listed FIRST so they shadow the C patterns in base_rewrite.
# Buffer convention LOCKED against mcc (mc task #2): global buffers are `PAddr`, accessed
# via `raw.load/store<T>(pa_offset(buf, idx*itemsize))` inside an `unsafe {}` block; reduce
# accumulators are local fixed arrays accessed with normal `[idx]` indexing.
# reg-ness is carried by addrspace (REG accumulators vs GLOBAL buffers), NOT by src op:
# acc reads index through an AFTER node, so an op-based check misses them.
def _is_reg(u): return u.addrspace == AddrSpace.REG

mc_rewrite = PatternMatcher([
  # reduce accumulator: local fixed array
  (UPat(Ops.DEFINE_REG, name="x"), lambda ctx,x: f"var {ctx[x]}: [{x.max_numel()}]{ctx.render_dtype(x.dtype.base)} = uninit;"),

  # indexing: REG accumulator -> `acc[idx]`; GLOBAL buffer -> byte address `pa_offset(buf, idx*size)`
  (UPat(Ops.INDEX, src=(UPat.var("buf"), UPat.var("idx")), name="x"), lambda ctx,x,buf,idx:
    f"{ctx[buf]}[{ctx[idx]}]" if _is_reg(x) else f"pa_offset({ctx[buf]}, ({ctx[idx]}) * {buf.dtype.base.itemsize})"),

  # load/store: REG reads/writes the array element directly; GLOBAL goes through raw.load/store<T>
  (UPat(Ops.LOAD, src=(UPat.var("bidx"),), name="x"), lambda ctx,x,bidx:
    f"{ctx[bidx]}" if _is_reg(bidx) else f"raw.load<{ctx.render_dtype(x.dtype)}>({ctx[bidx]})"),
  (UPat(Ops.STORE, src=(UPat.var("bidx"), UPat.var("var"))), lambda ctx,bidx,var:
    f"{ctx[bidx]} = {ctx[var]};" if _is_reg(bidx) else f"raw.store<{ctx.render_dtype(var.dtype)}>({ctx[bidx]}, {ctx[var]});"),

  # float const: C `0.0f` -> MC `0.0`
  (UPat(Ops.CONST, dtype=dtypes.float, name="x"), lambda ctx,x: f"{float(x.arg)}"),

  # select: MC `if` is a STATEMENT, not an expression, so `let x = if..` won't parse.
  # Lower to a pure helper call (a call IS an expression). Helper emitted by render_kernel.
  (UPat(Ops.WHERE, name="x"), lambda ctx,x: f"select_{ctx.render_dtype(x.dtype)}({ctx[x.src[0]]}, {ctx[x.src[1]]}, {ctx[x.src[2]]})"),

  # MC rejects negative float literals and `a + -c`. tinygrad lowers `x - c` to `x + (-c)`,
  # so rewrite ADD-with-negative-float-const back into subtraction by a positive literal.
  (UPat(Ops.ADD, src=(UPat.var("a"), UPat(Ops.CONST, dtype=dtypes.float, name="c"))),
   lambda ctx,a,c: f"({ctx[a]} - {-float(c.arg)})" if c.arg < 0 else None),
  (UPat(Ops.ADD, src=(UPat(Ops.CONST, dtype=dtypes.float, name="c"), UPat.var("a"))),
   lambda ctx,a,c: f"({ctx[a]} - {-float(c.arg)})" if c.arg < 0 else None),
]) + base_rewrite   # ALU / CMP / etc. infix forms are identical in MC, reuse them

class MCRenderer(ClangRenderer):
  # Subclass ClangRenderer to inherit its CPU execution-model flags (has_local, global_max,
  # etc.) so lowering produces the same plain RANGE loops as the C oracle. Force
  # single-threaded for Phase 1 (no SPECIAL/workitems) -> simplest sequential kernels.
  has_threads = False
  string_rewrite = mc_rewrite
  # NOTE: dtypes.int -> "usize" (not i32): in Phase-1 f32 kernels, ints only appear as loop
  # indices / index arithmetic, and MC forbids implicit int<->usize conversion, so unifying
  # all index math to usize keeps `data[alu]` well-typed. Revisit when porting integer
  # *data* dtypes (then index-vs-data int needs distinguishing). See finding #1.
  type_map = {dtypes.float: "f32", dtypes.float64: "f64", dtypes.int: "usize", dtypes.uint32: "u32",
              dtypes.bool: "bool", dtypes.uint64: "usize", dtypes.int64: "i64"}
  infinity = "INF"   # TODO: from mc mathf surface
  nan = "NAN"        # TODO: from mc mathf surface

  def render_dtype(self, dt:DType, mutable=True) -> str:
    if isinstance(dt, PtrDType):
      return "PAddr"   # global buffers are passed as physical addresses (see buffer convention)
    return self.type_map.get(dt.scalar(), dt.scalar().name)

  def render_kernel(self, function_name:str, kernel:list[str], bufs, uops, prefix=None) -> str:
    # MC: `export fn NAME(name: PAddr, ...) -> void { unsafe { ... } }`
    params = []
    for name,(u,mutable) in bufs:
      if isinstance(u.dtype, PtrDType): t = "PAddr"
      elif u.dtype == dtypes.int: t = "usize"
      else: t = self.render_dtype(u.dtype)
      params.append(f"{name}: {t}")
    body = "\n".join("  "+ln for ln in kernel)  # extra indent: body lives in an unsafe block
    # emit a pure select helper per dtype used by WHERE (MC `if` is a statement, not an expr)
    helpers = ""
    for dt in dedup_dtypes(w.dtype for w in uops if w.op is Ops.WHERE):
      ty = self.render_dtype(dt)
      helpers += (f"fn select_{ty}(c: bool, a: {ty}, b: {ty}) -> {ty} {{\n"
                  f"  if c {{ return a; }} else {{ return b; }}\n}}\n\n")
    return (f'import "std/addr.mc";\n\n{helpers}'
            f"export fn {function_name}({', '.join(params)}) -> void {{\n"
            f"  unsafe {{\n{body}\n  }}\n}}")

  def _render(self, uops:list[UOp]) -> tuple[str, list[str], list]:
    # MC-flavored copy of CStyleLanguage._render: `let/var name: T = expr;` decls and
    # while-loops (init before, increment at close) instead of C `for`.
    r: dict[UOp, str] = {}
    self.r = r
    from collections import Counter
    child_count = Counter(v for ru in uops for v in ru.src)
    writable_params = {u for u in UOp.sink(*[u.src[0] for u in uops if u.op is Ops.STORE]).toposort(lambda u: u.op != Ops.END)
                       if u.op is Ops.PARAM}
    bufs: dict[UOp, tuple] = {}
    kernel: list[str] = []
    depth = 1
    range_stack: list[str] = []
    c: defaultdict[str, int] = defaultdict(int)
    name = "test"
    for u in uops:
      if u.op in {Ops.NOOP, Ops.GROUP}: continue
      if u.op is Ops.AFTER: r[u] = r[u.src[0]]; continue
      if u.op is Ops.SINK:
        if u.arg is not None: name = u.arg.function_name
        continue
      if u.op in (Ops.PARAM, Ops.DEFINE_VAR):
        if u.op is not Ops.PARAM: r[u] = u.arg[0]
        else: r[u] = f"data{u.arg.slot}_{sz}" if (sz:=u.max_numel()) > 0 else f"data{u.arg.slot}"
        bufs[u] = (r[u], (u, u in writable_params))
        continue

      # --- MC while-loop handling (replaces C for) ---
      if u.op is Ops.RANGE:
        idx = f"{axis_letters[u.arg[-1]]}idx"+range_str(u)
        r[u] = idx
        bound = self[u.src[0]]
        kernel.append("  "*depth + f"var {idx}: usize = 0;")
        kernel.append("  "*depth + f"while {idx} < {bound} {{")
        range_stack.append(idx)
        depth += 1
        continue
      if u.op is Ops.END:
        idx = range_stack.pop()
        kernel.append("  "*depth + f"{idx} = {idx} + 1;")
        depth -= 1
        kernel.append("  "*depth + "}")
        continue
      if u.op is Ops.ENDIF:
        depth -= 1
        kernel.append("  "*depth + "}")
        continue

      # naming
      prefix = None
      if u.op is Ops.SPECIAL: r[u] = u.arg
      else:
        prefix = {Ops.DEFINE_LOCAL: "temp", Ops.CONST: "const", Ops.CAST: "cast", Ops.BITCAST: "cast",
                  Ops.GEP: "gep", Ops.INDEX: "bidx", Ops.DEFINE_REG: "acc", Ops.LOAD: "val"}.get(u.op, "alu")
        r[u] = f"{prefix}{c[prefix]}"

      l = cast(str, self.string_rewrite.rewrite(u, ctx=self))
      assert l is not None, f"failed to render {u.op} {u.dtype} {u.arg}"

      if u.op is Ops.IF:
        kernel.append("  "*depth + f"if {self[u.src[0]]} {{"); depth += 1; continue

      # inline single-use pure expressions; otherwise emit a `let` binding
      if (u.op in {Ops.CONST, Ops.GEP, Ops.INDEX} or
          (u.op is Ops.LOAD and u.src[0].addrspace == AddrSpace.REG) or
          (u.op in {*(GroupOp.ALU-{Ops.WHERE}), Ops.CAST, Ops.BITCAST} and child_count[u] == 1)):
        r[u] = l
      else:
        if u.op not in {Ops.DEFINE_REG, Ops.STORE} and u.dtype != dtypes.void:
          l = f"let {r[u]}: {self.render_dtype(u.dtype)} = {l};"
        kernel.append("  "*depth + l)
        if prefix: c[prefix] += 1
    del self.r
    return (name, kernel, list(bufs.values()))
