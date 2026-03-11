from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class AppConfig:
    raw: dict[str, Any]

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AppConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        return cls(raw=data)

    def __getitem__(self, item: str) -> Any:
        return self.raw[item]

    def get(self, item: str, default: Any = None) -> Any:
        return self.raw.get(item, default)

    @property
    def output_root(self) -> Path:
        return Path(self.raw.get("output_root", "outputs"))

    @property
    def image_size(self) -> int:
        return int(self.raw["data"]["image_size"])

    @property
    def seed(self) -> int:
        return int(self.raw.get("seed", 42))
