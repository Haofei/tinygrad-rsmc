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
RENDER = os.path.join(HERE, "render_kernel.rss")

def rss_render():
  env = dict(os.environ, RSSCRIPT_RUNTIME_PATH=os.path.join(RSS_ROOT, "crates/runtime"))
  p = subprocess.run([RSS, "run", RENDER], capture_output=True, env=env, cwd=RSS_ROOT)
  if p.returncode != 0:
    print("rss run failed:\n", p.stderr.decode()); sys.exit(1)
  return p.stdout.decode().strip() + "\n"

def main():
  mc_kernel = rss_render()
  print("=== rss-rendered MC kernel ===\n" + mc_kernel)
  # the rss kernel is `export fn E_8(...)`; harness() drops the kernel's own import line.
  hsrc = harness(mc_kernel, "E_8", n_in=2, n_out=8, ksize=8)
  a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
  b = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]
  expected = [x + y for x, y in zip(a, b)]
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
    stdin = b"".join(struct.pack("<f", v) for v in a + b)
    p = subprocess.run([os.path.join(w, "k")], input=stdin, capture_output=True)
    got = [struct.unpack("<f", p.stdout[i:i+4])[0] for i in range(0, len(p.stdout), 4)]
  ok = got == expected
  print(f"\n{'PASS' if ok else 'FAIL'} rss-frontend e2e: got {got}, want {expected}")
  sys.exit(0 if ok else 1)

if __name__ == "__main__":
  main()
