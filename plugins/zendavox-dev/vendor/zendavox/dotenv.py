"""Tiny, dependency-free ``.env`` loader.

Avoids pulling in python-dotenv for one small need. Reads simple ``KEY=VALUE``
lines and sets them in the environment without overriding variables that are
already set. Lines that are blank, commented, or malformed are ignored.
"""

from __future__ import annotations

import os
import pathlib


def load_env(path: str | os.PathLike[str] = ".env") -> None:
    file = pathlib.Path(path)
    if not file.is_file():
        return
    for raw_line in file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
