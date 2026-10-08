"""The checks script ``scripts/checks/api_export.py`` as a module, for the export tests."""

import importlib.util
from types import ModuleType

from tests.conftest import REPO


def load_script() -> ModuleType:
    path = REPO / "scripts" / "checks" / "api_export.py"
    spec = importlib.util.spec_from_file_location("api_export_check", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
