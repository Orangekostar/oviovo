#!/usr/bin/env bash
set -euo pipefail

compiler="${OVIMAP_CXX_REAL:-/home/ww/miniconda3/envs/ovimap-cropformer/bin/x86_64-conda-linux-gnu-g++}"
arguments=()
for argument in "$@"; do
    if [[ "$argument" == "-Wl,--sysroot=/" ]]; then
        continue
    fi
    arguments+=("$argument")
done

exec "$compiler" "${arguments[@]}"
