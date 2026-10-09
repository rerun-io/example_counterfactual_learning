#!/usr/bin/env bash
# Start the open-source Rerun catalog server over rerun/ and keep it running (Ctrl-C to stop).
# Every dataset, layer, table and blueprint is registered from the files under rerun/ on start, so restarting is safe.
# Open rerun+http://127.0.0.1:51234 in the viewer (`rerun rerun+http://127.0.0.1:51234`).
source "$(cd "$(dirname "$0")" && pwd)/lib.sh"
mkdir -p "$ACT_RERUN_DIR"
exec python -m cfnbc.serve --port 51234
