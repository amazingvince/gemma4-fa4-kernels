# CLAUDE.md

Read `AGENTS.md`, then `docs/model-contract.md` and `docs/status.md`.

For any CuTe DSL design, implementation, port, review, debugging, or tuning
work, read the full `skills/writing-cute-dsl-kernels/SKILL.md` router and the
references it requires. Project routing lives in
`skills/gemma4-kernel-project/SKILL.md`.

Never model Gemma 4 global attention as `K is V` at the FMHA boundary, and
never let FA4 choose its generic `1/sqrt(d)` default: the model scale is 1.0.
