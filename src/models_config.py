from __future__ import annotations

from pathlib import Path
from typing import Any, List

import yaml


def load_models_from_yaml(path: str | Path) -> List[dict[str, Any]]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    models = data.get("models")
    if not isinstance(models, list):
        raise ValueError(f"{p}: expected top-level 'models' list")
    return models
