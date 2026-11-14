#!/usr/bin/env bash
set -euo pipefail
exec /opt/inferboost-ui/bin/python /opt/pipeline/image/entry.py "$@"
