import json
from pathlib import Path


class Logger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.data = {}

    def log(self, key: str, value) -> None:
        if key not in self.data:
            self.data[key] = []
        self.data[key].append(value)

    def flush(self) -> None:
        self.path.write_text(json.dumps(self.data))