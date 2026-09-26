"""The one thing the hooks and the MCP server have to agree on.

A coding session is not one process. The session-start hook runs, exits, and is
gone; the MCP server runs for the length of the session; the session-end hook
is a third process again. They all need the same answer to one question: which
Zendavox session is this work being recorded under?

So it is kept in a small file, outside any repository - a checkout is the wrong
place for something this transient, and something transient is the wrong thing
to risk committing.

**It is keyed on the checkout, not on the harness's session id.** The obvious
choice is the session id, and it is wrong: the hooks are handed one on stdin
and the MCP server is not, so the two halves would key on different things and
never see each other's state. The symptom is quiet and expensive - work
recorded fine, and a session record left open forever, because the end hook
looked somewhere the server never wrote. The checkout is the one identifier
both halves always have.

The cost of that choice is that two sessions open in the same checkout share a
session record. That is the right trade: they are two sittings at the same
project, which is what a session record describes anyway, and the alternative
was a mechanism that did not work at all.

The Zendavox session is opened **lazily**, by the first write, not by the start
hook. Opening one at every session start would file a session record for every
time someone opened a terminal and closed it again, and a record full of empty
sessions is worse than no record: it makes the real ones hard to find.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import re
import tempfile
from dataclasses import dataclass

from zendavox.dev.config import HOME_DIR_NAME

STATE_DIR_NAME = "dev-sessions"

#: A key is used to build a filename, so it is whitelisted rather than trusted.
_SAFE = re.compile(r"[^A-Za-z0-9._-]")

#: Keep the readable part short enough that the hash still fits comfortably.
_READABLE = 60

#: When there is nothing at all to key on.
FALLBACK_KEY = "unknown-project"


@dataclass(frozen=True)
class SessionState:
    """What a coding session has recorded so far."""

    session_id: str | None = None
    title: str | None = None
    closed: bool = False


def state_dir() -> pathlib.Path:
    return pathlib.Path.home() / HOME_DIR_NAME / STATE_DIR_NAME


def key_for_root(root: pathlib.Path | str | None) -> str:
    """The state key for a checkout.

    Readable enough to identify by eye in the state directory, and suffixed
    with a hash of the full path so that two long paths sharing a prefix - or
    a path trimmed to fit - cannot collide onto one another's session.
    """
    if root is None:
        return FALLBACK_KEY
    # A Path and an equivalent string must hash identically, or a hook (which
    # may receive either) and the MCP server land on different files for the
    # same checkout. str() on a Path is platform-native - backslashes on
    # Windows - while a plain string argument is used exactly as given, so the
    # two diverge on Windows for the same logical path. as_posix() is stable
    # across platforms and matches a forward-slash string unchanged.
    full = (root.as_posix() if isinstance(root, pathlib.Path) else str(root)).strip()
    if not full:
        return FALLBACK_KEY
    digest = hashlib.sha256(full.encode("utf-8")).hexdigest()[:12]
    readable = _SAFE.sub("_", full).strip("_")[-_READABLE:].strip("_")
    return f"{readable}-{digest}" if readable else digest


def path_for(key: str, *, directory: pathlib.Path | None = None) -> pathlib.Path:
    base = state_dir() if directory is None else directory
    return base / f"{key}.json"


def read(key: str, *, directory: pathlib.Path | None = None) -> SessionState:
    """What we know about work in this checkout. A missing or damaged file
    means "nothing recorded yet", which is the truth and is also safe."""
    try:
        raw = path_for(key, directory=directory).read_text(encoding="utf-8")
        parsed = json.loads(raw)
    except (OSError, json.JSONDecodeError):
        return SessionState()
    if not isinstance(parsed, dict):
        return SessionState()
    session_id = parsed.get("session_id")
    title = parsed.get("title")
    return SessionState(
        session_id=session_id if isinstance(session_id, str) else None,
        title=title if isinstance(title, str) else None,
        closed=bool(parsed.get("closed")),
    )


def write(
    key: str,
    state: SessionState,
    *,
    directory: pathlib.Path | None = None,
) -> None:
    """Record the state, atomically, and never fatally.

    Atomically because the MCP server and an end hook can write at the same
    moment, and a half-written file would read as a lost session. Never
    fatally because a full disk is not a reason to fail somebody's turn.
    """
    target = path_for(key, directory=directory)
    payload = json.dumps(
        {
            "session_id": state.session_id,
            "title": state.title,
            "closed": state.closed,
        }
    )
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        handle, temp_name = tempfile.mkstemp(
            dir=str(target.parent), prefix=".zdx-", suffix=".tmp"
        )
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as out:
                out.write(payload)
            os.replace(temp_name, target)
        except OSError:
            _remove(pathlib.Path(temp_name))
            raise
    except OSError:
        return


def clear(key: str, *, directory: pathlib.Path | None = None) -> None:
    _remove(path_for(key, directory=directory))


def _remove(path: pathlib.Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        return
