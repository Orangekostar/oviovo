"""Load unchanged released evaluators with their own package-relative imports."""
import hashlib
import importlib.util
from pathlib import Path
import sys
import types


def load_released_module(path):
    path = Path(path).resolve()
    package_name = '_ovimap_release_' + hashlib.sha256(str(path.parent).encode()).hexdigest()[:16]
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [str(path.parent)]
        package.__package__ = package_name
        sys.modules[package_name] = package
    name = package_name + '.' + path.stem
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return vars(module)
