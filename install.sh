#!/usr/bin/env bash
set -euo pipefail
command -v python3 >/dev/null || { echo 'InferBoost 安装需要主机 Python 3.10+。' >&2; exit 1; }
exec python3 "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/installer/runtime.py" install "$@"
