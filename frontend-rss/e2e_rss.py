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

def compile_run(mc_kernel, inputs, n_in):
  hsrc = harness(mc_kernel, "E_8", n_in=n_in, n_out=8, ksize=8)
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

def check(label, mc_kernel, inputs, n_in, expected):
  got = compile_run(mc_kernel, inputs, n_in)
  ok = got == expected
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
  kernels = rss_render("uop_render.rss").split("//---KERNEL---")
  k_add, k_mul, k_relu = [k.strip() + "\n" for k in kernels]
  ok &= check("arena/add",  k_add,  [a, b], 2, [x + y for x, y in zip(a, b)])
  ok &= check("arena/mul",  k_mul,  [a, b], 2, [x * y for x, y in zip(a, b)])
  ok &= check("arena/relu", k_relu, [c],    1, [max(0.0, x) for x in c])
  sys.exit(0 if ok else 1)
