# Third-party and source provenance

Transformers source is not copied; setup checks out its pinned revision.

The repository carries one focused patch against the pinned FlashAttention
CuTe interface for the exact H100 asymmetric d512-QK/d256-V forward shape.
The base revision and patch hash are recorded in `upstream.lock.json`. The
patched upstream file retains its original copyright header, and the
FlashAttention BSD-3-Clause license is reproduced at
`third_party/flash-attention/LICENSE`.

The directory `skills/writing-cute-dsl-kernels/` is the user-supplied skill
package `writing-cute-dsl-kernels-v1.1.0`; its internal manifest, checksums,
merge notes, source map, and validation report are preserved unchanged.

The project reference code is an independent small-shape implementation of the
publicly documented/index-level model operation. Source revisions and URLs are
recorded in `upstream.lock.json` and `configs/model/gemma4-31b.lock.json`.
