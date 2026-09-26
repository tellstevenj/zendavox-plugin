"""``python -m zendavox.dev`` - the one entry point everything else names.

The hooks and the MCP server are configured by writing a command into a
settings file, and a command written into a settings file is a promise: it will
be run, unattended, by a harness that will not read an error message. So there
is exactly one of them, it takes a subcommand, and every subcommand that the
harness runs exits 0 no matter what happens inside it.

    check           - is this set up, and can it reach Zendavox? (for people)
    warm            - wake the service before reading or writing (for people
                      and for anything that hit a timeout mid-session)
    brief           - print the project's brief as text (for people)
    groups          - how chats are filed in the sidebar, and what that has
                      left waiting to be read into the record (for people)
    drift           - did a chat stay on the subject it started on? Reads
                      transcripts already on disk, writes nothing (for
                      people, and for setting the thresholds the product
                      will later run on)
    hook start      - session-start hook: emit the brief as opening context
    hook end        - session-end hook: close an open session record
    hook prompt     - prompt hook: warn when a chat drifts off its subject,
                      and past a limit hold the message back
    mcp             - run the MCP server on stdin and stdout

``check`` is the exception that may exit non-zero: it exists to be run by a
person who wants to know whether this works, and answering "fine" when it is
not would defeat the only purpose it has.
"""

from __future__ import annotations

import os
import pathlib
import sys

from zendavox.dev import drift, grouping, state
from zendavox.dev.brief_text import render
from zendavox.dev.client import Client, DevError
from zendavox.dev.config import (
    KEY_ENV,
    DevConfig,
    home_config_path,
    load,
    redact,
)
from zendavox.dev.hooks import (
    OUTPUT_ENV,
    prompt_submit,
    session_end,
    session_start,
)
from zendavox.dev.mcp import Server, serve
from zendavox.dev.preflight import summarise, warm

USAGE = """zendavox-dev - continuity between coding sessions.

  python -m zendavox.dev check          is this set up, and does it work?
  python -m zendavox.dev warm           wake the service, say which half is asleep
  python -m zendavox.dev brief          print this project's brief
  python -m zendavox.dev groups         how chats are filed, and what is waiting
  python -m zendavox.dev groups --backlog
                                        queue every chat already in a group
  python -m zendavox.dev groups --recorded local_<uuid>
                                        mark one thread as read into the record
  python -m zendavox.dev hook start     session-start hook (harness runs this)
  python -m zendavox.dev hook end       session-end hook (harness runs this)
  python -m zendavox.dev hook prompt    topic check before each message (harness)
  python -m zendavox.dev mcp            MCP server on stdin/stdout
"""


def _root() -> pathlib.Path:
    """The checkout being worked in.

    The harness sets ``CLAUDE_PROJECT_DIR``; falling back to the working
    directory keeps the command usable by hand.
    """
    told = os.environ.get("CLAUDE_PROJECT_DIR", "").strip()
    return pathlib.Path(told) if told else pathlib.Path.cwd()


def _client(config: DevConfig) -> Client | None:
    if not config.configured or config.key is None:
        return None
    return Client(
        base_url=config.base_url, key=config.key, timeout=config.timeout
    )


def _warm(config: DevConfig) -> int:
    """Wake the service and report which half, if either, is asleep.

    Three exit codes, so a script can branch without parsing English: 0 ready,
    1 not ready but worth retrying, 2 a person has to do something - a paused
    database, a refused key, no key at all.
    """
    print(f"Zendavox:  {config.base_url}")
    print(f"Key:       {redact(config.key)}")
    result = warm(config)
    print()
    print(summarise(result))
    if result.ok:
        return 0
    print()
    print(result.detail)
    if result.needs_a_person:
        print()
        print("Retrying will not fix this one.")
        return 2
    return 1


def _check(config: DevConfig) -> int:
    print(f"Zendavox:  {config.base_url}")
    print(f"Key:       {redact(config.key)}")
    print(f"Read from: {config.key_source or '(nowhere - none found)'}")

    if not config.configured:
        print()
        print("No key, so this session has no memory. To fix it:")
        print(f"  1. Sign in at {config.base_url}/connect and issue a key.")
        print("  2. Put it somewhere this can find it, either")
        print(f"       {KEY_ENV}=zdx_... in the environment or a .env file,")
        print(f"       or {{\"key\": \"zdx_...\"}} in {home_config_path()}")
        return 1

    line = _client(config)
    assert line is not None
    try:
        brief = line.brief()
    except DevError as exc:
        print()
        print(f"Could not reach the project: {exc}")
        return 1

    sessions = brief.get("session_count")
    print()
    print(f"Connected. {sessions} session(s) recorded for this project.")
    return 0


def _brief(config: DevConfig) -> int:
    line = _client(config)
    if line is None:
        print("No key configured; run 'check' to set one up.", file=sys.stderr)
        return 1
    try:
        print(render(line.brief()))
    except DevError as exc:
        print(f"Could not fetch the brief: {exc}", file=sys.stderr)
        return 1
    return 0


def _project_name(config: DevConfig) -> str | None:
    """Which Zendavox project this checkout's key writes to.

    Asked of the service rather than assumed, because the answer decides which
    waiting work this session is allowed to do, and guessing it wrong files
    somebody's history under the wrong customer.
    """
    line = _client(config)
    if line is None:
        return None
    try:
        name = line.brief().get("project_name")
    except DevError:
        return None
    return name if isinstance(name, str) and name.strip() else None


def _groups(config: DevConfig, args: list[str]) -> int:
    """Show how chats are filed, and what that has left waiting."""
    if "--recorded" in args:
        where = args.index("--recorded")
        if where + 1 >= len(args):
            print("Which chat? Give its local_<uuid>.", file=sys.stderr)
            return 2
        local_id = args[where + 1]
        done = False
        for workspace in grouping.survey():
            saved = grouping.read_state(workspace.scope)
            if grouping.mark_recorded(saved, local_id):
                grouping.write_state(workspace.scope, saved)
                done = True
        if done:
            print(f"{local_id} is off the queue and marked as read.")
            return 0
        print(f"{local_id} was not waiting on the queue.", file=sys.stderr)
        return 1

    project = _project_name(config)
    found = grouping.sweep(project=project, backlog="--backlog" in args)

    if not found.workspaces:
        print("No Claude Code sidebar on this machine, so nothing to watch.")
        return 0

    for workspace in found.workspaces:
        counts: dict[str, int] = {}
        for local_id in workspace.chats:
            group = workspace.group_of(local_id)
            if group:
                counts[group] = counts.get(group, 0) + 1
        print(f"Workspace {workspace.scope}")
        for name in workspace.groups.values():
            print(f"  {name:<24} {counts.get(name, 0)}")
        print(f"  {'(ungrouped)':<24} {len(workspace.ungrouped)}")
        print()

    print(f"This checkout records to: {project or '(unknown - no key, or no reply)'}")
    print(f"Waiting for this project:  {len(found.waiting)}")
    print(f"Queued just now:           {len(found.queued)}")
    print()
    text = grouping.describe(found, project=project)
    print(text if text else "Nothing to say about the sidebar.")
    return 0


def _number(args: list[str], flag: str, fallback: float) -> float:
    """One ``--flag value`` off the command line, or the default.

    A value that is not a number falls back rather than raising: this
    command is for looking at chats, and refusing to run over a typo would
    be the wrong trade.
    """
    if flag not in args:
        return fallback
    where = args.index(flag)
    if where + 1 >= len(args):
        return fallback
    try:
        return float(args[where + 1])
    except ValueError:
        return fallback


def _drift(args: list[str]) -> int:
    """Did chats stay on the subject they started on?

    Reads transcripts and nothing else. The thresholds are on the command
    line because finding the right ones is what this is for.
    """
    root = grouping.projects_dir()
    if "--in" in args:
        where = args.index("--in")
        if where + 1 < len(args):
            root = pathlib.Path(args[where + 1])
    if not root.is_dir():
        print(f"No transcripts at {root}.", file=sys.stderr)
        return 1

    warmup = int(_number(args, "--warmup", drift.WARMUP_TURNS))
    window = int(_number(args, "--window", drift.WINDOW))
    threshold = _number(args, "--threshold", drift.OFF_THRESHOLD)
    run_to_drift = int(_number(args, "--run", drift.RUN_TO_DRIFT))
    limit = int(_number(args, "--limit", 0))
    verbose = "--verbose" in args or "--turns" in args
    hold_away = "--hold-away" in args

    paths = list(drift.transcripts(root))
    if limit:
        paths = paths[:limit]
    readings = [
        drift.read(
            path,
            warmup=warmup,
            window=window,
            threshold=threshold,
            run_to_drift=run_to_drift,
            hold_away=hold_away,
        )
        for path in paths
    ]

    print(f"Reading {root}")
    print(
        f"Settings: warmup={warmup}, window={window}, "
        f"threshold={threshold}, run={run_to_drift}"
        + (", hold-away" if hold_away else "")
    )
    print()
    for reading in readings:
        if "--only-drifted" in args and reading.drifted_at is None:
            continue
        print(drift.describe(reading, verbose=verbose))
    print()
    print(drift.summarise(readings))
    ever = [r for r in readings if r.drifted_at is not None]
    back = [r for r in ever if r.returns]
    print(f"  {'ever went off':<14} {len(ever)}")
    print(f"  {'and came back':<14} {len(back)}")
    return 0


def _speak_utf8() -> None:
    """Make stdout carry any character the brief contains.

    The brief is prose written by people and by the product, and it holds
    en-dashes, arrows and whatever a customer typed into a fact. On Windows a
    process whose stdout is a pipe gets the system codepage, cp1252, and
    printing a single arrow raises UnicodeEncodeError halfway through - after
    part of the brief has already been written. The harness then reads a
    truncated brief, or none, and `zendavox-dev.cmd` exits 0 over the top of
    it, so nothing anywhere says a word.

    Found 2026-09-11 by running the hook by hand through a pipe. Everything
    else in this package is careful never to cost somebody their session; this
    was the one path that could, and it did it silently. ``errors="replace"``
    because a character that still will not encode should cost one question
    mark, never the brief.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except (AttributeError, ValueError, OSError):
            continue


def main(argv: list[str] | None = None) -> int:
    _speak_utf8()
    args = list(sys.argv[1:] if argv is None else argv)
    command = args[0] if args else ""

    if command == "mcp":
        # Config is resolved lazily inside the server: an MCP server that
        # refused to start without a key would show up as a broken connection
        # rather than as a tool that explains what is missing.
        return serve(Server(root=_root()))

    if command == "hook":
        which = args[1] if len(args) > 1 else ""
        raw = sys.stdin.read() if not sys.stdin.isatty() else ""
        if which == "start":
            text = session_start(
                raw,
                root=_root(),
                output_mode=os.environ.get(OUTPUT_ENV, "json").strip().lower(),
            )
            if text:
                print(text)
            return 0
        if which == "end":
            session_end(raw, root=_root())
            return 0
        if which == "prompt":
            text = prompt_submit(raw)
            if text:
                print(text)
            return 0
        # An unknown hook name must still not break a session.
        print(f"[zendavox-dev] unknown hook {which!r}", file=sys.stderr)
        return 0

    if command == "drift":
        # Handled before the config is loaded: this reads files on disk and
        # reaches neither the network nor a database, so it must work on a
        # machine that has no connector key at all.
        return _drift(args[1:])

    config = load(root=_root())
    if command == "warm":
        return _warm(config)
    if command == "check":
        return _check(config)
    if command == "brief":
        return _brief(config)
    if command == "groups":
        return _groups(config, args[1:])
    if command == "where":
        # Undocumented on purpose: prints where the session state lives, for
        # when a session's record is not where somebody expects it.
        print(state.state_dir())
        return 0

    print(USAGE, file=sys.stderr)
    return 0 if command in ("", "-h", "--help", "help") else 2


if __name__ == "__main__":
    raise SystemExit(main())
