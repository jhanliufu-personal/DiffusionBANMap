#!/bin/bash
# Runs all embedding jobs (resumable). On a GPU, raise N_INET (e.g. 20000) for a bigger ImageNet bank.
set -e
cd "$(dirname "$0")/../.."
L=outputs/bank_geometry/logs
uv run --with "rembg[cpu]" python -m analysis.bank_geometry.build_imagesets inet_cutout > $L/cutout.log 2>&1 &
CUT=$!
uv run python -m analysis.bank_geometry.build_imagesets edits > $L/edits.log 2>&1
uv run python -m analysis.bank_geometry.build_imagesets inet_raw > $L/inet_raw.log 2>&1
wait $CUT
uv run python -m analysis.bank_geometry.build_imagesets inet_styled > $L/inet_styled.log 2>&1
uv run python -m analysis.bank_geometry.build_imagesets inet_gray > $L/inet_gray.log 2>&1
echo ALL_EMBEDDINGS_DONE
