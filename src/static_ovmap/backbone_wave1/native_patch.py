"""Generate and verify the new patch relative to the existing capture layer."""

import os
from pathlib import Path
import subprocess
import tempfile

from .binding import read


def export_patch(spec, repo):
    repo, upstream = Path(repo), Path(spec["upstream_worktree"])
    destination = repo / "third_party_patches/ovimap/backbone_wave1_v1/backbone_wave1_v1.patch"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=spec["native_build_root"]) as temporary:
        env = dict(os.environ, GIT_INDEX_FILE=str(Path(temporary) / "index"))
        def git(*args, output=False):
            result = subprocess.run(["git", *args], cwd=upstream, env=env,
                                    stdout=subprocess.PIPE if output else None, check=True)
            return result.stdout
        git("read-tree", spec["upstream_commit"])
        git("apply", "--cached", str(repo / spec["base_native_patch"]))
        patch = git("diff", "--binary", output=True)
        destination.write_bytes(patch)
        git("apply", "--cached", "--check", str(destination))
        git("apply", "--cached", str(destination))
        git("diff", "--exit-code", "--quiet")
    return destination


if __name__ == "__main__":
    spec_path = Path(__file__).parents[3] / "docs/paper/static_ovmap/backbone_wave1_v1/PROTOCOL_SPEC.json"
    spec = read(spec_path)
    print(export_patch(spec, spec["worktree"]))
