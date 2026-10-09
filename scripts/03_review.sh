#!/usr/bin/env bash
# 2. Review: QA metrics as a `qa` layer + segment properties on every demo, registered with the running server, and a
# review checklist of viewer links (results/reviews/demo_review.md). CPU only.
source "$(cd "$(dirname "$0")" && pwd)/lib.sh"
require_server
python -m cfnbc.qa --define A50 --version 1     # also records the 50 demos (with file hashes) in table dataset_defs
python -m cfnbc.hub sync --datasets cube_demos
python -m cfnbc.review demos
echo "In the viewer: open cube_demos and sort the segment table by qa:lift_gap"
