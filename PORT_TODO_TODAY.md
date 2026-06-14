# Port Todo - 2026-06-14

Current portman snapshot:
- Overall symbol coverage: 51.4%
- Public API coverage: 55.2% of 2285 API symbols
- Weighted plan coverage: 43.7%
- UOp area coverage: 73.4%
- Implemented symbols: 1691
- Not started symbols: 1603
- Verified symbols: 1

Working tree note:
- Today has uncommitted provenance-header changes in `tinygrad-rss/src/uop/{ops,methods,symbolic,render,upat}.rss`.
- Those headers intentionally map split RSS files back to `tinygrad/uop/ops.py`.
- After adding the headers, several false UOp gaps disappeared from portman, including `UOp.const_factor`, `UOp.divides`, `UOp.divide_exact`, `UOp.is_increasing`, `UOp.pop_const`, `UOp.pyrender`, and `graph_rewrite`.

## Next Batch: UOp/UPat Public Surface

Goal: close the current high-risk `uop/ops.py` public gaps as a group, not one method at a time.

Real wrapper work:
- Add `UOp.alu` equivalent over the existing `mk_alu1`, `mk_alu2`, and `mk_alu3` helpers.
- Add `UOp.const` equivalent over `mk_const_int` and `mk_const_float`.
- Add `UOp.unique_const` equivalent over existing `UNIQUE`, `DEVICE`, and `CONST` machinery.
- Add `UOp.replace` equivalent using existing `uop_replace_arg` and `uop_replace_dtype` patterns.
- Ensure `UOp.shape` maps to existing `uop_shape` from `shape.rss`.
- Add `UPat.const`, `UPat.const_like`, `UPat.alu`, and `UPat.dtype` equivalents over existing `upat_*` helpers.

Likely alias or portman-link work:
- `TrackedPatternMatcher.rewrite` should map to the existing rewrite path.
- `UOp.index` should map to `uop_index`.
- `UOp.ranges` should map to `uop_ranges`.
- `UOp.reduce` should map to `uop_reduce`.
- `UOp.shard` should map to `uop_shard`.
- `UPat.after`, `UPat.end`, `UPat.index`, `UPat.reduce`, and `UPat.sink` should map to existing flattened helpers.
- `bitcast`, `ssimplify`, and `sym_infer` exist but may need explicit links if portman keeps them ambiguous.

Portman improvement to consider:
- Split-file provenance works, but broad split files can increase map time.
- Portman should probably support a `symbols:` hint or owner-prefix rule for split files so `uop/ops.py` can map cleanly without scoring unrelated candidates.
- Portman should expose an easier command to list ambiguous candidates for a gap, so we can decide alias-vs-real-port quickly.

## Follow-Up Batches

1. DType small gap batch
- Investigate `dtype.py::truncate`.
- Decide whether it is a true RSS type alias gap or a missing dtype helper.

2. Tensor training mode batch
- Port `Tensor.train` context/class behavior.
- Check whether RSS needs a better context-manager pattern or whether an explicit helper is enough.

3. Codegen exception/context batch
- Port lightweight missing classes: `CFGContext`, `LinearScanRegallocContext`, `KernelOptError`, `TimeoutException`.
- Treat type-only CUDA architecture constants separately.

4. Engine JIT control batch
- Port `DepsTracker`, `GraphException`, `JitError`, and `MultiGraphRunner`.
- Do not revive deleted/useless `engine` prototype files; work only in the mirrored `tinygrad-rss/src/engine` tree.

5. Runtime/generated batch
- Keep `runtime/autogen` vendored/copied, not hand-ported.
- Make sure portman continues excluding or separately tracking generated files.

## Verification Commands

Run after the UOp/UPat batch:

```sh
cd /home/zoe/portman
PYTHONPATH=src python3 -m portman inventory
PYTHONPATH=src python3 -m portman map
PYTHONPATH=src python3 -m portman status
PYTHONPATH=src python3 -m portman gaps --public
```

Then run the project verifier:

```sh
cd /home/zoe/tinygrad-rsmc
RSSCRIPT_RUNTIME_PATH=/home/zoe/rsscript/crates/runtime /home/zoe/rsscript/target/release/rss run tinygrad-rss
python3 oracle/roundtrip.py
```

## Stop Point

Paused here for today before adding the UOp/UPat wrapper batch. The next coding session should start by either:
- finishing the wrapper batch described above, or
- converting the alias-needed UOp/UPat gaps into explicit portman links if the code already exists.
