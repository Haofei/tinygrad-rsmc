"""
Phase-2 end-to-end proof: the rss FRONTEND renders an MC kernel, which the mc BACKEND
compiles and runs, and the result is checked numerically.

    python3 frontend-rss/e2e_rss.py

Pipeline: rss run render_kernel.rss -> MC source -> (harness) -> mcc emit-c --profile=hosted
-> clang -lm -> native -> feed inputs -> verify out == a+b.
Both halves of the port are now their own languages: frontend in rss, backend in mc.
"""
import os, sys, struct, subprocess, tempfile
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "oracle"))
from roundtrip import harness, MAIN_C, MC_ROOT, MCC  # reuse the proven harness/driver

RSS_ROOT = "/home/zoe/rsscript"
RSS = os.path.join(RSS_ROOT, "target/release/rss")

def rss_render(rss_file):
  env = dict(os.environ, RSSCRIPT_RUNTIME_PATH=os.path.join(RSS_ROOT, "crates/runtime"))
  p = subprocess.run([RSS, "run", os.path.join(HERE, rss_file)], capture_output=True, env=env, cwd=RSS_ROOT)
  if p.returncode != 0:
    print(f"rss run {rss_file} failed:\n", p.stderr.decode()); sys.exit(1)
  return p.stdout.decode().strip() + "\n"

def compile_run(mc_kernel, inputs, n_in, n_out=8, ksize=8):
  hsrc = harness(mc_kernel, "E_8", n_in=n_in, n_out=n_out, ksize=ksize)
  with tempfile.TemporaryDirectory() as w:
    open(os.path.join(w, "k.mc"), "w").write(hsrc)
    open(os.path.join(w, "main.c"), "w").write(MAIN_C)
    c = subprocess.run([MCC, "emit-c", os.path.join(w, "k.mc"), "--profile=hosted"],
                       capture_output=True, cwd=MC_ROOT)
    if c.returncode != 0:
      print("mcc failed:\n", c.stderr.decode()); sys.exit(1)
    open(os.path.join(w, "k.c"), "w").write(c.stdout.decode())
    cc = subprocess.run(["clang", "-std=c11", os.path.join(w, "k.c"), os.path.join(w, "main.c"),
                         "-lm", "-o", os.path.join(w, "k")], capture_output=True)
    if cc.returncode != 0:
      print("clang failed:\n", cc.stderr.decode()); sys.exit(1)
    stdin = b"".join(struct.pack("<f", v) for arr in inputs for v in arr)
    p = subprocess.run([os.path.join(w, "k")], input=stdin, capture_output=True)
    return [struct.unpack("<f", p.stdout[i:i+4])[0] for i in range(0, len(p.stdout), 4)]

def check(label, mc_kernel, inputs, n_in, expected, n_out=8, ksize=8):
  got = compile_run(mc_kernel, inputs, n_in, n_out, ksize)
  ok = all(abs(g - e) < 1e-4 for g, e in zip(got, expected)) and len(got) == len(expected)
  print(f"{'PASS' if ok else 'FAIL'} {label}: got {got}, want {expected}")
  return ok

def run_program(program, inputs):
  """Compile+run a COMPLETE hosted MC program (already has its own main); no harness."""
  with tempfile.TemporaryDirectory() as w:
    open(os.path.join(w, "k.mc"), "w").write(program)
    open(os.path.join(w, "main.c"), "w").write(MAIN_C)
    c = subprocess.run([MCC, "emit-c", os.path.join(w, "k.mc"), "--profile=hosted"], capture_output=True, cwd=MC_ROOT)
    if c.returncode != 0:
      print("mcc failed:\n", c.stderr.decode()[:800]); return None
    open(os.path.join(w, "k.c"), "w").write(c.stdout.decode())
    cc = subprocess.run(["clang", "-std=c11", os.path.join(w, "k.c"), os.path.join(w, "main.c"), "-lm", "-o", os.path.join(w, "k")], capture_output=True)
    if cc.returncode != 0:
      print("clang failed:\n", cc.stderr.decode()[:800]); return None
    stdin = b"".join(struct.pack("<f", v) for arr in inputs for v in arr)
    p = subprocess.run([os.path.join(w, "k")], input=stdin, capture_output=True)
    return [struct.unpack("<f", p.stdout[i:i+4])[0] for i in range(0, len(p.stdout), 4)]

def check_program(label, program, inputs, expected):
  got = run_program(program, inputs)
  ok = got is not None and len(got) == len(expected) and all(abs(g - e) < 1e-4 for g, e in zip(got, expected))
  print(f"{'PASS' if ok else 'FAIL'} {label}: got {got}, want {expected}")
  return ok

if __name__ == "__main__":
  a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
  b = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]
  c = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0]
  ok = True
  # parameterized template path
  ok &= check("template/add", rss_render("render_kernel.rss"), [a, b], 2, [x + y for x, y in zip(a, b)])
  # real arena-walker path: several ops from one rss program, split on the marker
  full = rss_render("uop_render.rss")
  kernels_section, rest = full.split("//---PROGRAM---")
  program, train_prog = rest.split("//---TRAIN---")
  program = program.strip() + "\n"
  train_prog = train_prog.strip() + "\n"
  k_add, k_mul, k_relu, k_sum, k_mm = [k.strip() + "\n" for k in kernels_section.split("//---KERNEL---")]
  ok &= check("arena/add",  k_add,  [a, b], 2, [x + y for x, y in zip(a, b)])
  ok &= check("arena/mul",  k_mul,  [a, b], 2, [x * y for x, y in zip(a, b)])
  ok &= check("arena/relu", k_relu, [c],    1, [max(0.0, x) for x in c])
  ok &= check("arena/sum",  k_sum,  [a],    1, [sum(a)], n_out=1)
  # matmul 4x4: ma @ mb (row-major, 16 elements each)
  ma = [float(i + 1) for i in range(16)]
  mb = [float((i * 7 + 3) % 5 - 2) for i in range(16)]  # non-trivial, exercises real accumulation
  W = 4
  mm_expected = [sum(ma[i * W + k] * mb[k * W + j] for k in range(W)) for i in range(W) for j in range(W)]
  ok &= check("arena/matmul", k_mm, [ma, mb], 2, mm_expected, n_out=16, ksize=16)

  # complete multi-kernel program (relu then add, shared buffer) generated by rss
  ok &= check_program("program/relu+add chain", program, [c, b], [max(0.0, x) + y for x, y in zip(c, b)])
  # end-to-end TRAINING: linear regression by SGD, fit t=2x+1 -> learned w~2, b~1
  ok &= check_program("train/linreg (SGD, fit 2x+1)", train_prog, [], [2.0, 1.0])
  sys.exit(0 if ok else 1)
