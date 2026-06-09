# Porting briefing for subagents (rss frontend / mc(C) backend)

Goal: faithfully port a tinygrad file into `/home/zoe/tinygrad-rsmc/port-rss/<name>.rss`,
validate it RUNS via the rss interpreter (and, for kernels, compiles+runs through mc), then
report. Source of truth = `/home/zoe/tinygrad` (locked v0.13.0-160). Validate behavior against
real tinygrad via `python3 -c "import tinygrad..."` (PYTHONPATH=/home/zoe/tinygrad).

## How to run an rss program (no rebuild needed; binary is prebuilt)
```
cd /home/zoe/rsscript
RSSCRIPT_RUNTIME_PATH=/home/zoe/rsscript/crates/runtime ./target/release/rss run <abs path>.rss
```
Filter noise: `... 2>&1 | grep -ivE "freshness|MVP|trusts|^\s*\||^\s*=|^\s*$|warning"`.
DO NOT rebuild rss or modify `/home/zoe/rsscript` (another agent owns compiler fixes).

## rss language idioms (FOLLOW THESE — they are the common failure causes)
- First line `features: local` IF you use `take`/`local` ownership; otherwise omit.
- **One statement per line.** No `;` to separate statements. `{ a() b() }` on one line = RS0015.
- Struct/sum: `struct S derives(Clone, Eq, Hash) { field: Type \n ... }`, `sum E derives(Clone,Eq,Hash){ A B C }`.
- Calls use named args: `List.push(list: mut xs, value: read v)`, `List.get(list: read xs, index: i)`.
- Effects on params are required: `read` (shared), `mut` (mutable), `take` (move/own). A param
  with no effect on a managed/sum type errors RS0008.
- Reassignment needs `let mut x = ...; x = ...`.
- `/` truncates toward zero (NOT Python floor). Implement floordiv explicitly if needed.
- Recursive managed structs work (e.g. `struct UOp { src: List<UOp> ... }`).
- Returning a fresh value: a function returning `fresh T` may return a constructor call, a call
  to another `fresh` fn, or a clean local bound to such. It may NOT return a local that was
  passed as `mut`/aliased → use an out-param instead: `fn f(..., out: mut List<T>)`.

## Known rss gaps + the WORKAROUNDS to use (do not fight these)
- **Building a node from `read` children works**: `fn uadd(a: read UOp, b: read UOp){ local s=List.new<UOp>(); List.push(list: mut s, value: read a); List.push(list: mut s, value: read b); return UOp(op: Add, src: take s, arg: 0) }`.
- **No generic `op: Ops` value param** (RS0008/awkward). Write one constructor per op (inline the
  variant literal), and dispatch with `if u.op == Add { ... }` chains, NOT a generic builder.
- **`==` on a `read` enum PARAM fails** (`&Ops==Ops`). Field access compares fine (`u.op == Add`).
  For a bare enum param use `match op { Add => {...} _ => {...} }`.
- **`List.get` on a struct-field `List<Int>` loses `Int` type** in `==`/arithmetic (returns T).
  Route through a typed helper: `fn iget(l: read List<Int>, i: Int) -> Int { return List.get(list: read l, index: i) }`.
- **No Int→Float conversion** (`Int.to_float`/`Float.from_int` don't resolve). If you need float
  consts, store them as `Float` literals (e.g. a `cval: Float` field), don't convert from Int.
- **Can't clone a `String`/managed field out of a `read` value** (RS1101 "cannot move out of ...
  behind shared reference"). To move a fresh value into a struct field, take ownership:
  `fn f(base: take T){ return Wrapper(field: take base) }` (needs `features: local`), and at the
  call site bind first: `let b = make(); f(base: take b)` (can't `take` an inline fresh expr).
  If you only have a `read`, build from literals instead of cloning.
- **Avoid `let mut s = <read-param>` then reassigning** (codegen E0308). Instead seed a fresh:
  `let mut s = String.concat(left: read label, right: read "")`.
- Strings: `String.concat(left: read a, right: read b)`, `String.from_int(value: n)`,
  `Int.to_string(value: read n)`, `String.from_float(value: f)`, `String.join(parts: read xs, separator: read "\n")`.
- Collections: `List.new<T>()`, `List.push/get/len/set/pop`, `Set.new<T>()`/`Set.insert/contains/len`,
  `Map.new<K,V>()`. Set/Map keys need `derives(..., Eq, Hash)`.
- Output: `Log.write(message: read s)`.

## mc kernel convention (for codegen/renderer ports that must EXECUTE)
Emit a hosted mc program (string), then validate with the harness below. Proven pattern:
```
import "std/addr.mc";
import "std/hosted_io.mc";
// optional: import "std/mathf.mc";  // exp_f32/log_f32/tanh_f32/sqrt_f32
fn select_f32(c: bool, a: f32, b: f32) -> f32 { if c { return a; } else { return b; } }
export fn E(data0: PAddr, data1: PAddr) -> void {
  unsafe {
    var i: usize = 0;
    while i < N {
      let xi: f32 = raw.load<f32>(pa_offset(data1, i * 4));   // BIND loads to a `let` (do NOT
      raw.store<f32>(pa_offset(data0, i * 4), <expr over xi>); //  nest raw.load inside the store
      i = i + 1;                                               //  arg — mc UnsupportedCEmission)
    }
  }
}
// + read_exact + hosted_kernel_run() that reads stdin into x[], calls E, writes o[] to stdout
```
- mc known bug (an agent is fixing it): a nested `raw.load(...)` inside a `raw.store(...)` arg
  fails. So always assign loads/intermediates to `let` temps first (this is also faithful: cstyle
  names every UOp to a variable).

## Validation harness (compile+run a rendered mc kernel and check numbers)
```python
import os,subprocess,struct,tempfile,sys
sys.path.insert(0,"/home/zoe/tinygrad-rsmc/oracle"); from roundtrip import MC_ROOT,MCC,MAIN_C
RSS="/home/zoe/rsscript"; env=dict(os.environ,RSSCRIPT_RUNTIME_PATH=RSS+"/crates/runtime")
# 1) run your .rss to emit the mc program text (Log.write the program)
prog=subprocess.run([RSS+"/target/release/rss","run","<file>.rss"],capture_output=True,env=env,cwd=RSS).stdout.decode().strip()+"\n"
with tempfile.TemporaryDirectory() as w:
    open(w+"/k.mc","w").write(prog); open(w+"/main.c","w").write(MAIN_C)
    subprocess.run([MCC,"emit-c",w+"/k.mc","--profile=hosted"],capture_output=True,cwd=MC_ROOT)  # -> k.c
    # clang -std=c11 k.c main.c -lm -o k ; feed stdin floats; read stdout floats
```
Pure-rss ports (dtype/uop/autodiff/etc.) just need `rss run` to print the expected results.

## Output / deliverable rules for subagents
- Write ONE new file `port-rss/<name>.rss`. Make it run clean and print validation lines that
  match tinygrad's actual values (compute the expected values from real tinygrad and compare).
- **Do NOT `git commit`** and do NOT touch other files — the parent process commits.
- Report: the file path, what tinygrad behavior it covers, and the validation output proving it.
