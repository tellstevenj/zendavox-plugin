"""Where a coding session gets its key, and which Zendavox it talks to.

Three sources, in falling order of precedence, each with a distinct job:

1. **The environment.** ``ZENDAVOX_DEV_KEY`` and ``ZENDAVOX_DEV_URL``. What a
   test, a one-off run, or a CI job sets to override everything below.
2. **The repository's ``.env``.** The convention this project already uses for
   the database URL, and already git-ignored. Right when the key belongs to
   one project and one checkout.
3. **``~/.zendavox/dev.json``.** Outside any repository, so one key serves
   every checkout on the machine. Right for the ordinary case: a person who
   wants their sessions to have continuity everywhere, not in one folder.

A key is a credential. It is read from git-ignored places on purpose, is never
written back anywhere, and :func:`redact` exists so diagnostics can say which
key is in play without printing it.
"""

from __future__ import annotations

import json
import os
import pathlib
from dataclasses import dataclass

from zendavox.dotenv import load_env

KEY_ENV = "ZENDAVOX_DEV_KEY"
URL_ENV = "ZENDAVOX_DEV_URL"
TIMEOUT_ENV = "ZENDAVOX_DEV_TIMEOUT"

#: Where the product actually lives. Overridable for local development.
DEFAULT_BASE_URL = "https://www.zendavox.com"

#: Long enough for a sleeping free-tier service to wake, short enough that a
#: session start is not held hostage by an outage. The hook is allowed to give
#: up: a session with no brief is worse than one with, and far better than one
#: that never starts.
DEFAULT_TIMEOUT = 12.0

#: The machine-wide home for settings that should outlive any one checkout.
HOME_DIR_NAME = ".zendavox"
HOME_CONFIG_NAME = "dev.json"


@dataclass(frozen=True)
class DevConfig:
    """How to reach Zendavox, and whether we can at all."""

    base_url: str
    key: str | None
    timeout: float
    #: Which of the three sources the key came from. Diagnostics only; when
    #: the wrong project's brief turns up, this is the first question.
    key_source: str | None

    @property
    def configured(self) -> bool:
        return bool(self.key)


def redact(key: str | None) -> str:
    """A key, shown the way the product shows it: enough to recognise, not
    enough to use. Mirrors ``api_key.key_prefix`` on the server side."""
    if not key:
        return "(none)"
    return f"{key[:11]}..." if len(key) > 14 else "(set)"


def home_config_path() -> pathlib.Path:
    return pathlib.Path.home() / HOME_DIR_NAME / HOME_CONFIG_NAME


def _read_home_config(path: pathlib.Path) -> dict[str, str]:
    """Read the machine-wide settings file, tolerating everything.

    A hand-edited JSON file with a trailing comma must not stop a session from
    starting, so a broken file reads as an absent one.
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    if not isinstance(parsed, dict):
        return {}
    return {
        str(k): str(v)
        for k, v in parsed.items()
        if isinstance(v, (str, int, float))
    }


def home_settings() -> dict[str, str]:
    """Every plain setting in ``~/.zendavox/dev.json``, for callers other
    than :func:`load` - the drift alerts read their on/off switch here."""
    return _read_home_config(home_config_path())


def _clean(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def load(
    *,
    root: pathlib.Path | None = None,
    environ: dict[str, str] | None = None,
    home_config: pathlib.Path | None = None,
) -> DevConfig:
    """Resolve the configuration for a coding session.

    ``root`` is the repository being worked in; its ``.env`` is consulted for a
    key. Everything is injectable so the resolution order can be tested without
    a real home directory or a real checkout.
    """
    env = os.environ if environ is None else environ

    key = _clean(env.get(KEY_ENV))
    url = _clean(env.get(URL_ENV))
    timeout_raw = _clean(env.get(TIMEOUT_ENV))
    source = "environment" if key else None

    # The repository's own .env. load_env never overrides what is already set,
    # so reading it cannot displace a deliberate choice made in the shell.
    if root is not None and environ is None:
        load_env(root / ".env")
        if key is None:
            key = _clean(os.environ.get(KEY_ENV))
            source = ".env" if key else None
        if url is None:
            url = _clean(os.environ.get(URL_ENV))
        if timeout_raw is None:
            timeout_raw = _clean(os.environ.get(TIMEOUT_ENV))

    if key is None or url is None or timeout_raw is None:
        path = home_config_path() if home_config is None else home_config
        settings = _read_home_config(path)
        if key is None:
            key = _clean(settings.get("key"))
            source = str(path) if key else None
        if url is None:
            url = _clean(settings.get("base_url"))
        if timeout_raw is None:
            timeout_raw = _clean(settings.get("timeout"))

    try:
        timeout = DEFAULT_TIMEOUT if timeout_raw is None else float(timeout_raw)
    except ValueError:
        # A mistyped timeout should not be fatal, and should not be zero -
        # either would turn a settings typo into "the hook never works".
        timeout = DEFAULT_TIMEOUT
    if timeout <= 0:
        timeout = DEFAULT_TIMEOUT

    return DevConfig(
        base_url=(url or DEFAULT_BASE_URL).rstrip("/"),
        key=key,
        timeout=timeout,
        key_source=source,
    )
