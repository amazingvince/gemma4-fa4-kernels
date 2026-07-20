# agent_space/ — scratch and retained hardware provenance

Lab notes, profiler dumps (*.ncu-rep), bench JSONL from in-progress runs,
repro scripts, and PTX/SASS dumps are disposable and Git-ignored. Named
`h100-check-*.json` files may be force-tracked as immutable environment and
patch-stack evidence for an accepted experiment; those files are
checksum-locked and must not be deleted. A minimal accepted-experiment bundle
under `remote-h100-expNNNN/` may also be force-tracked when its matrix,
sanitizer, or codegen evidence cannot be represented faithfully by the JSONL
ledger alone. Such bundles contain compact JSON/text evidence, are cited by the
experiment record, and are checksum-locked; raw profiler, PTX, cubin, SASS, and
cache dumps remain ignored. Other durable results graduate via the loop in
AGENTS.md: records -> experiments/, configs -> tuning/, code -> kernels/.
