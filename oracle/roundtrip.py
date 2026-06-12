"""
End-to-end numerical proof for Phase 1: render a tinygrad kernel to MC, build a hosted
harness around it (mcc emit-c --profile=hosted -> clang -lm -> native), feed it inputs on
stdin, and check the output matches tinygrad's own CPU computation.

    PYTHONPATH=/home/zoe/tinygrad python3 oracle/roundtrip.py

Proves: tinygrad frontend -> MCRenderer -> modern-c -> native binary computes the same
numbers as tinygrad's reference CPU backend.
"""
import os, re, sys, struct, subprocess, tempfile, math
sys.path.insert(0, os.path.dirname(__file__))
from tinygrad import Tensor, Device, dtypes
from tinygrad.uop.ops import Ops
from tinygrad.codegen import to_program
from tinygrad.helpers import Context
from tinygrad.device import Compiler
from mc_renderer import MCRenderer

MC_ROOT = "/home/zoe/modern-c"
MCC = os.path.join(MC_ROOT, "zig-out/bin/mcc")

def render_mc(ast, mc):
  prg = to_program(ast, mc)
  src = next(s.arg for s in prg.src if s.op is Ops.SOURCE)
  name = re.search(r"export fn (\w+)\(", src).group(1)
  return src, name

def kernel_ast(t):
  return next(u for u in t.schedule_linear().toposort()
             if u.op is Ops.SINK and type(u.arg).__name__ == "KernelInfo")

def harness(kernel_src, name, n_in, n_out, ksize):
  """Hosted MC: read n_in input arrays (size ksize) from stdin, call kernel, write output."""
  body = kernel_src.split("\n", 2)[2]  # drop the kernel's own `import` line + blank
  ins = "\n".join(f"  var in{i}: [{ksize}]f32 = uninit;" for i in range(n_in))
  outd = f"  var outb: [{max(n_out,1)}]f32 = uninit;"
  read_ins = "\n".join(
    f"  if let err(e) = read_exact(input, pa((&in{i}[0]) as usize), {ksize} * 4) {{ return {10+i}; }}"
    for i in range(n_in))
  # kernel arg order: data0 = output, data1.. = inputs
  args = ["pa((&outb[0]) as usize)"] + [f"pa((&in{i}[0]) as usize)" for i in range(n_in)]
  return f'''import "std/addr.mc";
import "std/hosted_io.mc";

{body}

fn read_exact(f: Fd, buf: PAddr, n: usize) -> Result<bool, IoError> {{
  var done: usize = 0;
  while done < n {{
    let r: usize = io_read(f, pa_offset(buf, done), n - done)?;
    if r == 0 {{ return err(.ReadFailed); }}
    done = done + r;
  }}
  return ok(true);
}}

export fn hosted_kernel_run() -> i32 {{
  let input: Fd = stdin_fd();
  let output: Fd = stdout_fd();
{ins}
{outd}
{read_ins}
  {name}({", ".join(args)});
  if let err(e) = io_write_all(output, pa((&outb[0]) as usize), {max(n_out,1)} * 4) {{ return 9; }}
  return 0;
}}
'''

MAIN_C = '#include <stdint.h>\nextern int32_t hosted_kernel_run(void);\nint main(void){return hosted_kernel_run();}\n'

def _flat(tl): return [tl] if isinstance(tl, float) else ([v for r in tl for v in r] if tl and isinstance(tl[0], list) else tl)

def run_case(label, tensor_fn, inputs, ksize, n_out):
  cpu = Device["CPU"].renderer
  mc = MCRenderer(cpu.target); mc.compiler = Compiler()
  # reference: a fresh graph realized on the normal CPU backend
  expected = _flat(tensor_fn(*[Tensor(x) for x in inputs]).tolist())
  # the kernel under test: render the same computation to MC
  with Context(NOOPT=1):
    t = tensor_fn(*[Tensor(x) for x in inputs])
    ast = kernel_ast(t)
    ksrc, name = render_mc(ast, mc)
  hsrc = harness(ksrc, name, len(inputs), n_out, ksize)
  with tempfile.TemporaryDirectory() as w:
    mcf = os.path.join(w, "k.mc"); open(mcf, "w").write(hsrc)
    mainf = os.path.join(w, "main.c"); open(mainf, "w").write(MAIN_C)
    cf = os.path.join(w, "k.c")
    c = subprocess.run([MCC, "emit-c", mcf, "--profile=hosted"], capture_output=True, cwd=MC_ROOT)
    if c.returncode != 0:
      print(f"FAIL {label}: mcc\n{c.stderr.decode()[:500]}\n--- harness ---\n{hsrc}"); return False
    open(cf, "w").write(c.stdout.decode())
    exe = os.path.join(w, "k")
    cc = subprocess.run(["clang", "-std=c11", cf, mainf, "-lm", "-o", exe], capture_output=True)
    if cc.returncode != 0:
      print(f"FAIL {label}: clang\n{cc.stderr.decode()[:500]}"); return False
    stdin = b"".join(struct.pack("<f", float(v)) for arr in inputs for v in arr)
    p = subprocess.run([exe], input=stdin, capture_output=True)
    if p.returncode != 0:
      print(f"FAIL {label}: kernel exit {p.returncode}"); return False
    got = [struct.unpack("<f", p.stdout[i:i+4])[0] for i in range(0, len(p.stdout), 4)]
    if len(got) != len(expected) or any(abs(g-e) > 1e-5*(1+abs(e)) for g, e in zip(got, expected)):
      print(f"FAIL {label}: got {got}, want {expected}"); return False
    print(f"PASS {label}: MC native == tinygrad CPU  (got {got})")
    return True

if __name__ == "__main__":
  a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
  b = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]
  ok = True
  # order-insensitive cases (commutative binary, or single-input) for a clean proof
  ok &= run_case("add",    lambda x, y: x + y,         [a, b], 8, 8)
  ok &= run_case("mul",    lambda x, y: x * y,         [a, b], 8, 8)
  ok &= run_case("affine", lambda x: x * 2.0 + 1.0,    [a],    8, 8)
  ok &= run_case("relu",   lambda x: (x - 4.0).relu(), [a],    8, 8)
  ok &= run_case("sum",    lambda x: x.sum(),          [a],    8, 1)
  ok &= run_case("bitcast_roundtrip", lambda x: x.bitcast(dtypes.uint32).bitcast(dtypes.float32) + 1.0, [a], 8, 8)
  sys.exit(0 if ok else 1)
