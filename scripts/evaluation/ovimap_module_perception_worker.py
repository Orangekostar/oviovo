"""Bind the unchanged native perception worker to the frozen local FP32 model."""

import functools
import os
import runpy
import sys
from pathlib import Path


def main():
    upstream = Path(os.environ["OVIMAP_NATIVE_UPSTREAM"])
    model = Path(os.environ["OVIMAP_NATIVE_MODEL"])
    if not (model / "config.json").is_file():
        raise FileNotFoundError(model)
    scripts = upstream / "scripts"
    sys.path.insert(0, str(scripts))
    import vl_models

    vl_models.siglip_model_list["siglip-l-16-384"] = str(model)
    vl_models.VLModel = functools.partial(vl_models.VLModel, precision="fp32")
    sys.argv.extend(["--scripts-dir", str(scripts)])
    runpy.run_path(str(scripts / "perception_worker.py"), run_name="__main__")


if __name__ == "__main__":
    main()
