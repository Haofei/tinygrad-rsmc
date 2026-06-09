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

# MC-specific per-UOp text. Listed FIRST so they shadow the C patterns in base_rewrite.
mc_rewrite = PatternMatcher([
  # reduce accumulator: C `float acc0[1];` -> MC `var acc0: [1]f32;`
  (UPat(Ops.DEFINE_REG, name="x"), lambda ctx,x: f"var {ctx[x]}: [{x.max_numel()}]{ctx.render_dtype(x.dtype.base)};"),

  # indexing: C `(buf+idx)` -> MC `buf[idx]`  (lvalue; LOAD/STORE use it directly)
  (UPat.var("buf").index(UPat.var("idx")), lambda ctx,buf,idx: f"{ctx[buf]}[{ctx[idx]}]"),

  # load/store: the index expr is already the element lvalue in MC
  (UPat(Ops.LOAD, src=(UPat.var("bidx"),)), lambda ctx,bidx: f"{ctx[bidx]}"),
  (UPat(Ops.STORE, src=(UPat.var("bidx"), UPat.var("var"))), lambda ctx,bidx,var: f"{ctx[bidx]} = {ctx[var]};"),

  # float const: C `0.0f` -> MC `0.0`
  (UPat(Ops.CONST, dtype=dtypes.float, name="x"), lambda ctx,x: f"{float(x.arg)}"),

  # select: C ternary `(c?a:b)` -> MC if-expression. TODO(mc-task#2): verify the exact
  # expression-level conditional syntax against mcc; demos show statement `switch`/`if`,
  # expression form assumed Zig-like here.
  (UPat(Ops.WHERE, name="x"), lambda ctx,x: f"(if {ctx[x.src[0]]} {{ {ctx[x.src[1]]} }} else {{ {ctx[x.src[2]]} }})"),
]) + base_rewrite   # ALU / CMP / etc. infix forms are identical in MC, reuse them

class MCRenderer(ClangRenderer):
  # Subclass ClangRenderer to inherit its CPU execution-model flags (has_local, global_max,
  # etc.) so lowering produces the same plain RANGE loops as the C oracle. Force
  # single-threaded for Phase 1 (no SPECIAL/workitems) -> simplest sequential kernels.
  has_threads = False
  string_rewrite = mc_rewrite
  type_map = {dtypes.float: "f32", dtypes.float64: "f64", dtypes.int: "i32", dtypes.uint32: "u32",
              dtypes.bool: "bool", dtypes.uint64: "usize", dtypes.int64: "i64"}
  infinity = "INF"   # TODO: from mc mathf surface
  nan = "NAN"        # TODO: from mc mathf surface

  def render_dtype(self, dt:DType, mutable=True) -> str:
    if isinstance(dt, PtrDType):
      return f"*mut {self.render_dtype(dt.base)}"
    return self.type_map.get(dt.scalar(), dt.scalar().name)

  def render_kernel(self, function_name:str, kernel:list[str], bufs, uops, prefix=None) -> str:
    # MC: `export fn NAME(name: type, ...) -> void { ... }`
    params = []
    for name,(u,mutable) in bufs:
      if isinstance(u.dtype, PtrDType): t = self.render_dtype(u.dtype, mutable)
      elif u.dtype == dtypes.int: t = "i32"
      else: t = self.render_dtype(u.dtype)
      params.append(f"{name}: {t}")
    body = "\n".join(kernel)
    return f"export fn {function_name}({', '.join(params)}) -> void {{\n{body}\n}}"

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
