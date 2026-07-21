# H100 all-layer compiled dispatch design brief

EXP-0042 widens only the accepted guarded, no-cache, tensor-only compiled
facade from the representative layer indices 0/local and 5/global to every
locked Gemma 4 text layer. It does not change a GPU kernel or admit raw
`torch.compile(layer)`.

## Scope and invariants

- H100/SM90, BF16, B1, inference/no-grad, text-only, no cache, zero-based
  positions, and `1 <= S <= 1024` remain unchanged.
- All 60 indices must match the locked family map: 50 local layers and 10
  global layers at indices 5, 11, ..., 59.
- Each facade captures its construction-time layer index. A later mutation to
  any other index, including another index in the same family, must fail before
  compiled entry.
- The validator requires the pinned terminal-family marker exactly: only layer
  58/local and layer 59/global have `store_full_length_kv=True`. With the locked
  `num_kv_shared_layers=0`, no later layer consumes those diagnostic stores and
  the tensor-only facade has no shared-KV side effect to reproduce.
- Live config, module, scalar, weight, storage, and shape validation remains on
  every call outside Dynamo.

## Single structural change

Parameterize the existing whole-layer validator by an explicit expected layer
index. The guarded facade passes its captured construction index; historical
raw-compiler and StaticCache callers continue to pass their existing narrow
indices. The tensor-only inner functions remain family-specific and contain no
Python module state.

## Frozen gates

- CPU/FakeTensor construction across indices 0..59 and same-family mutation
  rejection before compiled entry.
- H100 eager and Inductor facade calls for all 60 indices at S1, compared with
  each actual pinned layer's eager result.
- Representative local/global S33 and S1024 regression calls retain the
  EXP-0023 graph, stream, and numerical contracts.
- Exactly two FA4 forward application classes across the all-layer sweep: one
  local and one global; layer index must not enter the compile or FA4 key.
- The exact EXP-0041 FlashAttention patch stack, full local/H100 tests, pinned
  Transformers oracle, and bundle verifier remain green.

## Rollback

Reverting the project integration change restores the layer-0/layer-5 facade.
No upstream patch or kernel rollback is required.
