#!/bin/sh
# CI and local development share the locked project interpreter and command table.
set -eu
root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
if [ ! -x "$root/.venv/bin/python" ]; then
  printf '%s\n' 'Prepare the environment: env -u VIRTUAL_ENV uv sync --locked --extra dev' >&2
  exit 2
fi
exec "$root/.venv/bin/python" "$root/scripts/gate.py" "$@"
