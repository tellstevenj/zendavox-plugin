"""What the harness runs, on its own, without being asked.

This is the half of zendavox-dev that does not depend on anybody remembering
anything. :mod:`zendavox.connector` states the problem plainly - *if recording
depends on remembering to record, nothing gets recorded* - and the same is true
of reading. An assistant that has to decide to ask for context has already lost
the sessions where it did not think to.

So the read is a hook. The harness runs it before the first prompt, every time,
whether or not any model judged it worthwhile.

``session_start``
    Fetch the project's brief and hand it to the session as opening context.

``session_end``
    If the session recorded anything, make sure it is closed with a summary,
    so an interrupted session still leaves a readable trace instead of an open
    record nobody will ever close.

``prompt_submit``
    Before each message is sent, check whether the chat is drifting off the
    subject it started on; warn on screen, and past a limit hold the message
    back. See :mod:`zendavox.dev.drift_guard`.

**Failing soft is a hard requirement here.** Every path through this module
ends in exit code 0. A missing key, a sleeping host, a changed reply shape -
none of them are worth costing somebody their session. The worst outcome
allowed is a session that starts with no memory, which is exactly the session
they had before this package existed.
"""

from __future__ import annotations

import json
import pathlib
import sys
from typing import Any

from zendavox.dev import drift, drift_guard, grouping, state
from zendavox.dev.brief_text import render
from zendavox.dev.client import Client, DevError
from zendavox.dev.config import DevConfig, load, redact
from zendavox.dev.preflight import HOOK_DEADLINE, summarise, warm

#: Emit structured hook output by default. Set ``ZENDAVOX_DEV_HOOK_OUTPUT`` to
#: ``text`` for a harness that takes a hook's plain stdout as context instead;
#: that way a contract change is a settings edit, not a code change.
OUTPUT_ENV = "ZENDAVOX_DEV_HOOK_OUTPUT"

#: What we call a session we had to close on the session's behalf.
UNATTENDED_SUMMARY = (
    "Session ended without a summary of its own; closed by the "
    "zendavox-dev session-end hook so the record does not stay open."
)


def _note(message: str) -> None:
    """Say something to the operator, never to the session.

    Anything written to stdout by a hook is either parsed as a directive or
    read by the model, so diagnostics have exactly one safe channel.
    """
    print(f"[zendavox-dev] {message}", file=sys.stderr, flush=True)


def read_hook_input(raw: str) -> dict[str, Any]:
    """Parse what the harness sends on stdin, tolerating anything."""
    try:
        parsed = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _string(payload: dict[str, Any], key: str) -> str | None:
    value = payload.get(key)
    return value if isinstance(value, str) and value.strip() else None


def context_payload(text: str) -> str:
    """Wrap rendered context the way the harness expects to receive it."""
    return json.dumps(
        {
            "hookSpecificOutput": {
                "hookEventName": "SessionStart",
                "additionalContext": text,
            }
        }
    )


def _client(config: DevConfig) -> Client | None:
    if not config.configured or config.key is None:
        return None
    return Client(
        base_url=config.base_url, key=config.key, timeout=config.timeout
    )


def session_start(
    raw_input: str,
    *,
    root: pathlib.Path | None = None,
    output_mode: str = "json",
    config: DevConfig | None = None,
    client: Client | None = None,
) -> str:
    """Produce the opening context for a session. Returns what to print.

    An empty string means "say nothing", which is the right answer whenever
    Zendavox cannot be reached: a session with no brief is ordinary, and a
    session interrupted by an error message about a brief is not.
    """
    payload = read_hook_input(raw_input)
    where = root if root is not None else _cwd_of(payload)

    settings = load(root=where) if config is None else config
    line = _client(settings) if client is None else client
    if line is None:
        _note(
            "no key configured, so this session starts with no memory. "
            "Run 'python -m zendavox.dev check' to set one up."
        )
        return ""

    # Wake the service before asking it for anything. Free hosting sleeps in
    # two places and they fail differently: the web service boots on traffic
    # and only needs waiting for, while a paused database has to be restored
    # by hand. A bare brief() call cannot tell those apart, and until
    # 2026-09-11 reported both as "did not answer in time" - which reads as
    # "the record is unreachable" when the record is perfectly healthy.
    #
    # The probes go down this client's own transport, and on a shorter deadline
    # than the command-line default: a harness kills a hook that overruns, and
    # a killed hook says nothing at all.
    ready = warm(settings, client=line, deadline=HOOK_DEADLINE)
    _note(summarise(ready))
    if not ready.ok:
        if ready.needs_a_person:
            _note(
                "this one will not fix itself by retrying - see the line "
                "above for what needs doing."
            )
        return ""

    try:
        brief = line.brief()
    except DevError as exc:
        _note(f"could not fetch the brief ({exc}); starting without it.")
        return ""
    except Exception as exc:  # noqa: BLE001 - a hook may not raise, ever
        _note(f"unexpected failure fetching the brief: {exc!r}")
        return ""

    text = render(brief)
    _note(
        f"brief loaded using key {redact(settings.key)} "
        f"from {settings.key_source or 'an unnamed source'}."
    )

    filing = _filing(brief)
    if filing:
        text = f"{text}\n\n{filing}"

    return text if output_mode == "text" else context_payload(text)


def _filing(brief: dict[str, Any]) -> str:
    """What the sidebar says about chats filed into this project's groups.

    A chat is filed into a group long after it ran, and at that moment the
    thread becomes history belonging to a named project. Nothing else in the
    system notices, so it is noticed here: the hook already runs before every
    first prompt, so the check costs no new process and nothing to remember.

    The hook cannot read a thread and judge what in it was a decision, a fact
    or a finding - that needs a model. So it detects and hands the work over,
    the same division the brief itself already uses.

    Wrapped whole, because this is a convenience bolted onto the one thing that
    actually matters here. A sidebar that cannot be read is a reason to say
    nothing about the sidebar, never a reason to cost somebody their brief.
    """
    name = brief.get("project_name")
    project = name if isinstance(name, str) and name.strip() else None
    try:
        return grouping.describe(grouping.sweep(project=project), project=project)
    except Exception as exc:  # noqa: BLE001 - a hook may not raise, ever
        _note(f"could not check how chats are filed: {exc!r}")
        return ""


def session_end(
    raw_input: str,
    *,
    root: pathlib.Path | None = None,
    directory: pathlib.Path | None = None,
    config: DevConfig | None = None,
    client: Client | None = None,
) -> None:
    """Close this session's record when possible, preserving unswept work.

    The hook cannot inspect the conversation, so the connector rejects an
    unattended close without a sweep. Keeping the local state lets a later
    session find and backfill the interrupted work instead of claiming it was
    reviewed.
    """
    payload = read_hook_input(raw_input)
    where = root if root is not None else _cwd_of(payload)
    # Keyed on the checkout, because the MCP server that opened this record was
    # never told the harness's session id. See zendavox.dev.state.
    key = state.key_for_root(where)
    current = state.read(key, directory=directory)

    if current.session_id is None or current.closed:
        # Nothing was recorded, or the session already closed itself with a
        # real summary. Either way there is nothing worth doing here.
        state.clear(key, directory=directory)
        return

    settings = load(root=where) if config is None else config
    line = _client(settings) if client is None else client
    if line is None:
        return

    try:
        line.close_session(
            session_id=current.session_id, summary=UNATTENDED_SUMMARY
        )
        _note("closed the open session record.")
    except DevError as exc:
        _note(f"could not close the session record ({exc}).")
    except Exception as exc:  # noqa: BLE001 - a hook may not raise, ever
        _note(f"unexpected failure closing the session record: {exc!r}")
    else:
        state.clear(key, directory=directory)


def prompt_submit(raw_input: str, *, setting: str | None = None) -> str:
    """Check a message against the chat's subject before it is sent.

    Returns what to print: empty to let it through with nothing said, or the
    alert or the stop from :mod:`zendavox.dev.drift_guard`. Needs no key and
    no network - the transcript the harness names is the whole input - so it
    runs on a machine that has never been connected to Zendavox.
    """
    payload = read_hook_input(raw_input)
    prompt = _string(payload, "prompt")
    if prompt is None:
        return ""
    try:
        told = _string(payload, "transcript_path")
        path = pathlib.Path(told) if told else None
        turns, reply = drift.conversation(path) if path else ((), "")
        verdict = drift_guard.assess(
            turns,
            prompt,
            reply=reply,
            setting=drift_guard.mode() if setting is None else setting,
            path=path,
        )
        return drift_guard.hook_output(verdict)
    except Exception as exc:  # noqa: BLE001 - a hook may not raise, ever
        _note(f"topic check skipped: {exc!r}")
        return ""


def _cwd_of(payload: dict[str, Any]) -> pathlib.Path:
    """Which checkout this session is in, according to the harness."""
    told = _string(payload, "cwd")
    return pathlib.Path(told) if told else pathlib.Path.cwd()
