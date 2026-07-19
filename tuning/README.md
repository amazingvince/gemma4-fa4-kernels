# Tuning tables

Tuning keys include architecture, kernel family, dtype, mask class, GQA packing
factor, mode, and shape regime. SM90, SM100, and SM103 entries never silently
share constants.

A table entry is promoted only after correctness, sanitizer, generated-code,
and benchmark evidence is linked to an experiment ID. Runtime tensors and
actual per-step sequence values do not belong in compile or tuning keys.
