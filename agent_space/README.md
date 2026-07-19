# agent_space/ — scratch and retained hardware provenance

Lab notes, profiler dumps (*.ncu-rep), bench JSONL from in-progress runs,
repro scripts, and PTX/SASS dumps are disposable and Git-ignored. Named
`h100-check-*.json` files may be force-tracked as immutable environment and
patch-stack evidence for an accepted experiment; those files are
checksum-locked and must not be deleted. Other durable results graduate via
the loop in AGENTS.md: records -> experiments/, configs -> tuning/, code ->
kernels/.
