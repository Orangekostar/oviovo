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

source_records=()
clone_exact() {
  local name=$1
  local url=$2
  local commit=$3
  local target=$4
  git clone --filter=blob:none "$url" "$target"
  git -C "$target" checkout --detach "$commit"
  source_records+=("$name" "$commit" "$url")
}

clone_exact OVI-MAP "$ovimap_url" "$ovimap_commit" "$temporary_root/OVI-MAP"
clone_exact Entity "$entity_url" "$entity_commit" "$temporary_root/Entity"
clone_exact detectron2 "$detectron2_url" "$detectron2_commit" "$temporary_root/detectron2"

ros_source_root="$temporary_root/OVI-MAP/mapping_ros_ws/src"
clone_exact approxmvbb_catkin https://github.com/ethz-asl/approxmvbb_catkin.git b2e87786e77b8c9ce71e56ce2a1146bd32b7c68c "$ros_source_root/approxmvbb_catkin"
clone_exact catkin_boost_python_buildtool https://github.com/ethz-asl/catkin_boost_python_buildtool c8ea6e25bc5d98772d57b2ffd7619973952a6839 "$ros_source_root/catkin_boost_python_buildtool"
clone_exact catkin_simple https://github.com/catkin/catkin_simple.git 0e62848b12da76c8cc58a1add42b4f894d1ac21e "$ros_source_root/catkin_simple"
clone_exact eigen_catkin https://github.com/ethz-asl/eigen_catkin.git 3323b388540fa95ec9da6f9cd887f70ead055edb "$ros_source_root/eigen_catkin"
clone_exact eigen_checks https://github.com/ethz-asl/eigen_checks.git 22a6247a3df11bc285d43d1a030f4e874a413997 "$ros_source_root/eigen_checks"
clone_exact gflags_catkin https://github.com/ethz-asl/gflags_catkin.git fc38fc525f7d48881aebb27a7b9978453556bbd4 "$ros_source_root/gflags_catkin"
clone_exact glog_catkin https://github.com/ethz-asl/glog_catkin.git 40a9edadd15c59f8b57dc947d0135b0a007ea10b "$ros_source_root/glog_catkin"
clone_exact minkindr https://github.com/ethz-asl/minkindr.git 564f12639a8447d4d3e5e7707851424302941056 "$ros_source_root/minkindr"
clone_exact minkindr_ros https://github.com/ethz-asl/minkindr_ros.git 5528b042124fe056a7cf53f96c8b39e1e32ec2b9 "$ros_source_root/minkindr_ros"
clone_exact numpy_eigen https://github.com/ethz-asl/numpy_eigen.git f63b1dfe3b0a1ee21138caa1dcedd32c7f0411d9 "$ros_source_root/numpy_eigen"
clone_exact protobuf_catkin https://github.com/ethz-asl/protobuf_catkin 721a6cc17e7e937e7accfb88a9967b26134db6f9 "$ros_source_root/protobuf_catkin"
clone_exact vpp_msgs https://github.com/ethz-asl/vpp_msgs.git 6aff4c33a0d79536dd769d176ee5cd1285004c88 "$ros_source_root/vpp_msgs"

for environment in ovimap-cropformer ovimap-map; do
  output="$temporary_root/environment/$environment"
  mkdir -p "$output"
  conda list --name "$environment" --explicit > "$output/conda-explicit.txt"
  conda run --name "$environment" python -m pip freeze > "$output/pip-freeze.txt"
done

python3 - "$temporary_root/environment/source-hashes.json" \
  "${source_records[@]}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
records = sys.argv[2:]
if len(records) % 3:
    raise ValueError("source records must contain name, commit, and URL triples")
payload = {
    name: {"commit": commit, "url": url}
    for name, commit, url in zip(records[0::3], records[1::3], records[2::3], strict=True)
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
