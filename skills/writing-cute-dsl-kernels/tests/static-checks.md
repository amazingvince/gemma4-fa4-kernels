# Static Validation Expectations

The included validator checks package structure and authoring hygiene. It does not compile or execute a GPU kernel.

It verifies:

- `SKILL.md` frontmatter has exactly `name` and `description`;
- skill name matches directory and uses lowercase letters/numbers/hyphens;
- description begins with `Use when`;
- required references, templates, tests, source map, agent-development guide, and exemplar map exist;
- required architecture, exemplar-first, agentic workflow, SM103, SM120, CLC, and low-level escape-hatch terms are present;
- relative Markdown links resolve;
- non-template files contain no unfinished-work markers;
- official NVIDIA documentation/repository and selected production/evaluation source domains appear in the source map;
- JSON manifest is valid;
- Python validator itself compiles.

Runtime evaluation still requires:

- fresh coding-agent runs using `tests/evaluation-prompts.md`;
- an environment with a supported NVIDIA GPU;
- a matching CuTe DSL installation;
- compilation and correctness tests against the target revision;
- sanitizer and performance tools.
