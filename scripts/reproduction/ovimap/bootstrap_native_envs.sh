#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 BUILD_ROOT" >&2
  exit 2
fi

build_root=$1
if [[ -e "$build_root" ]]; then
  echo "build root already exists: $build_root" >&2
  exit 1
fi

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd -P)
parent=$(dirname "$build_root")
temporary_root="${build_root}.tmp.$$"
created_envs=()

cleanup() {
  status=$?
  if [[ $status -ne 0 ]]; then
    for environment in "${created_envs[@]}"; do
      conda env remove --name "$environment" --yes >/dev/null 2>&1 || true
    done
  fi
  if [[ -d "$temporary_root" ]]; then
    rm -rf "$temporary_root"
  fi
  exit "$status"
}
trap cleanup EXIT

for environment in ovimap-cropformer ovimap-map; do
  if conda env list | awk '{print $1}' | grep -Fxq "$environment"; then
    echo "conda environment already exists: $environment" >&2
    exit 1
  fi
done

mkdir -p "$parent" "$temporary_root/environment"

conda env create --file "$repo_root/configs/environments/ovimap_cropformer.yaml"
created_envs+=(ovimap-cropformer)
conda env create --file "$repo_root/configs/environments/ovimap_map.yaml"
created_envs+=(ovimap-map)

ovimap_commit=58a804e2d7c82ba05a489eb071aba3367301fed8
entity_commit=6e7e13ac91ef508088e1b848167c01f19b00b512
detectron2_commit=d1e04565d3bec8719335b88be9e9b961bf3ec464
ovimap_url=https://github.com/OVI-MAP/OVI-MAP.git
entity_url=https://github.com/qqlu/Entity.git
detectron2_url=https://github.com/facebookresearch/detectron2.git

git clone --filter=blob:none "$ovimap_url" "$temporary_root/OVI-MAP"
git -C "$temporary_root/OVI-MAP" checkout --detach "$ovimap_commit"
git clone --filter=blob:none "$entity_url" "$temporary_root/Entity"
git -C "$temporary_root/Entity" checkout --detach "$entity_commit"
git clone --filter=blob:none "$detectron2_url" "$temporary_root/detectron2"
git -C "$temporary_root/detectron2" checkout --detach "$detectron2_commit"

for environment in ovimap-cropformer ovimap-map; do
  output="$temporary_root/environment/$environment"
  mkdir -p "$output"
  conda list --name "$environment" --explicit > "$output/conda-explicit.txt"
  conda run --name "$environment" python -m pip freeze > "$output/pip-freeze.txt"
done

python3 - "$temporary_root/environment/source-hashes.json" \
  "$ovimap_commit" "$entity_commit" "$detectron2_commit" \
  "$ovimap_url" "$entity_url" "$detectron2_url" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
commits = sys.argv[2:5]
urls = sys.argv[5:8]
names = ("OVI-MAP", "Entity", "detectron2")
payload = {
    name: {"commit": commit, "url": url}
    for name, commit, url in zip(names, commits, urls, strict=True)
}
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

nvidia-smi -q > "$temporary_root/environment/nvidia-smi.txt"
python3 - "$temporary_root/environment/host.json" <<'PY'
import json
import platform
import sys
from pathlib import Path

os_release = {}
for line in Path("/etc/os-release").read_text(encoding="utf-8").splitlines():
    if "=" in line:
        key, value = line.split("=", 1)
        os_release[key] = value.strip('"')
payload = {
    "architecture": platform.machine(),
    "kernel": platform.release(),
    "os": os_release,
    "python": sys.version,
}
Path(sys.argv[1]).write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY

mv "$temporary_root" "$build_root"
trap - EXIT
