#!/usr/bin/env bash
set -euo pipefail
if [[ "$#" == 0 || "${1:-}" == idle ]]; then
  if [[ "${1:-}" != idle && -t 0 && -t 1 ]]; then
    exec /bin/bash
  fi
  echo 'InferBoost 容器已就绪。进入终端运行 quickstart；当前未启动模型HTTP服务。'
  exec sleep infinity
fi
exec "$@"
