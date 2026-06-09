"""
Render a set of tinygrad kernels to both C (reference) and MC (Phase-1 MCRenderer),
writing each to oracle/{c_reference,mc_generated}/. Run from the repo root:

    PYTHONPATH=/home/zoe/tinygrad python3 oracle/gen_oracle.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from tinygrad import Tensor, Device
from tinygrad.uop.ops import Ops
from tinygrad.codegen import to_program
from tinygrad.helpers import Context
from tinygrad.device import Compiler
from mc_renderer import MCRenderer

HERE = os.path.dirname(__file__)
C_DIR = os.path.join(HERE, "c_reference")
MC_DIR = os.path.join(HERE, "mc_generated")
os.makedirs(MC_DIR, exist_ok=True)

def kernel_asts(t: Tensor):
  lin = t.schedule_linear()
  return [u for u in lin.toposort() if u.op is Ops.SINK and type(u.arg).__name__ == "KernelInfo"]

def render(ast, renderer):
  prg = to_program(ast, renderer)
  return next(s.arg for s in prg.src if s.op is Ops.SOURCE)

CASES = {
  "add":   lambda: Tensor.empty(8) + Tensor.empty(8),
  "mul":   lambda: Tensor.empty(8) * Tensor.empty(8),
  "axpy":  lambda: Tensor.empty(8) * Tensor.empty(8) + Tensor.empty(8),
  "sum":   lambda: Tensor.empty(8).sum(),
  "relu":  lambda: Tensor.empty(8).relu(),
}

def main():
  cpu = Device["CPU"].renderer
  mc = MCRenderer(cpu.target)
  mc.compiler = Compiler()  # no-op: we only want rendered SOURCE, not a compiled binary
  with Context(NOOPT=1):
    for cname, build in CASES.items():
      asts = kernel_asts(build())
      for i, ast in enumerate(asts):
        tag = cname if len(asts) == 1 else f"{cname}_{i}"
        c_src = render(ast, cpu)
        mc_src = render(ast, mc)
        open(os.path.join(C_DIR, f"{tag}.c"), "w").write(c_src.strip()+"\n")
        open(os.path.join(MC_DIR, f"{tag}.mc"), "w").write(mc_src.strip()+"\n")
        print(f"==== {tag} ====")
        print("--- C ---");  print(c_src.strip())
        print("--- MC ---"); print(mc_src.strip())
        print()

if __name__ == "__main__":
  main()
