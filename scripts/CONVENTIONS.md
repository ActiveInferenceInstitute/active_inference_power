# Script conventions

Use `pathlib.Path`, explicit argument parsing, deterministic configuration, and
non-zero exit codes on failed gates. Scripts may perform file I/O and call the
controlled release infrastructure, but they must not duplicate correction,
simulation, power, or agent logic.

Resolve relative paths against the project root, validate an output root before
writing, and make every state-changing destination visible in the CLI. A
read-only audit must not repair or rewrite a candidate. Isolated diagnostics
must require a fresh explicit root; release and campaign producers must retain
their declared provenance/manifest boundaries rather than sharing a convenient
scratch directory.

Public commands need stable `--help` text and a documented producer/consumer
role in `README.md` and the output inventory. Supporting modules are not
alternate command-line release paths.
