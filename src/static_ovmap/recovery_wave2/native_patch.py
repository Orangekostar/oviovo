"""Recovery patch exported relative to the two already specified patch layers."""

import os
from pathlib import Path
import subprocess
import tempfile


def export_patch(repo, upstream, build_root):
    repo, upstream = Path(repo), Path(upstream)
    destination = repo / "third_party_patches/ovimap/recovery_wave2_v1/recovery_wave2_v1.patch"
    destination.parent.mkdir(parents=True, exist_ok=True)
    Path(build_root).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=build_root) as temporary:
        env = dict(os.environ, GIT_INDEX_FILE=str(Path(temporary) / "index"))
        def git(*argv, output=False):
            result = subprocess.run(["git", *argv], cwd=upstream, env=env, check=True,
                                    stdout=subprocess.PIPE if output else None)
            return result.stdout
        git("read-tree", "f8f7bcd0ca8228f6b8b4064f2e29dcee3a502424")
        for layer in ("module_validation_v1/ovimap_module_validation_v1.patch",
                      "backbone_wave1_v1/backbone_wave1_v1.patch"):
            git("apply", "--cached", str(repo / "third_party_patches/ovimap" / layer))
        patch = git("diff", "--binary", output=True)
        destination.write_bytes(patch)
        git("apply", "--cached", "--check", str(destination))
        git("apply", "--cached", str(destination))
        git("diff", "--exit-code", "--quiet")
    return destination


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--build-root", required=True)
    args = parser.parse_args()
    print(export_patch(Path(__file__).parents[3], args.upstream, args.build_root))
