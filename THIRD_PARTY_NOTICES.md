# Third-party and source provenance

This scaffold does not copy FlashAttention or Transformers source code. It
checks out pinned revisions during setup. Those projects retain their own
licenses and notices.

The directory `skills/writing-cute-dsl-kernels/` is the user-supplied skill
package `writing-cute-dsl-kernels-v1.1.0`; its internal manifest, checksums,
merge notes, source map, and validation report are preserved unchanged.

The project reference code is an independent small-shape implementation of the
publicly documented/index-level model operation. Source revisions and URLs are
recorded in `upstream.lock.json` and `configs/model/gemma4-31b.lock.json`.
