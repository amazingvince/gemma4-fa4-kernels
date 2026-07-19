# Agent prompt sequence

Use prompts in order. Each prompt deliberately has a hard scope boundary so a
coding agent cannot skip the model contract or benchmark ruler and jump
straight to an unreviewable kernel rewrite.

1. `00-bootstrap-and-m0.md` — primary starter prompt; establish the contract,
   pinned environment, both remotes, and baseline artifacts. No kernel edits.
2. `10-m0-baseline.md` — one-host baseline capture after bootstrap.
3. `20-b300-local-first-kernel.md` — first correctness-only kernel task.
4. `30-global-d512-design.md` — architecture design spike before code.
5. `40-transformers-integration.md` — explicit HF per-layer/multimodal adapter
   work after kernel interfaces exist.

Replace angle-bracket profile/experiment values, but do not weaken the listed
semantic invariants or evidence gates.
