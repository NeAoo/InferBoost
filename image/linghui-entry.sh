#!/usr/bin/env bash
set -euo pipefail
if [[ "$#" -gt 0 ]]; then exec "$@"; fi
if [[ -n "${EMP_NUMBER:-}" ]]; then
  printf '%s\n' "$EMP_NUMBER" > /home/jovyan/.empcode
fi
exec /usr/bin/code-server \
  --auth=none --bind-addr=0.0.0.0:5000 \
  --disable-update-check --disable-workspace-trust --disable-telemetry \
  --extensions-dir=/opt/inferboost-vscode-extensions /home/jovyan
