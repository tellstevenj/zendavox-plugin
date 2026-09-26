"""Topic drift, acted on: the alerts a person sees, and the stop.

:mod:`zendavox.dev.drift` measures whether a chat stayed on the subject it
started on. This module decides what to do about it, one message at a time,
before the message is sent. Steven, 2026-09-26: *"when the conversation
starts to get too far off track of the original conversation it provides
some kind of alerts that come up on the screen that warn the user about the
fact that the conversation is drifting and at some point stops the
conversation."*

**Three steps, each louder than the last,** keyed to how many messages *in a
row* have been away from the subject. The measurement found that nine times
out of ten a chat that dips away comes back (docs/CONTEXT_DRIFT.md), so one
stray message says nothing:

``notice``
    A quiet line on screen. Nothing else changes.
``warning``
    A plain warning with a countdown, and a note to the assistant so it can
    offer to take the new subject to a new chat.
``stop``
    The message is held back, not sent, and the person is told why and what
    to do: start a new chat, or resend with ``continue:`` in front, which
    makes the new subject this chat's subject from that message on.

A message that comes back to the subject resets the count, so a chat that
wanders and returns is never stopped.

**The stop needs a second opinion.** Word overlap is good enough to warn on
and not good enough to stop on. Run over 82 real chats on 2026-09-26, the
count reached the stop in 4, and reading them by hand, one had changed
subject - the person had typed "New subject" - and three had not. So before
a message is held back, a small model reads the opening and the latest
messages and says whether the chat has really left its subject
(:func:`ask_model`). When no model can be reached - Claude Code not signed
in, no API key - the chat is warned and never stopped. A warning that is
wrong costs a glance; a stop that is wrong costs the person their train of
thought.

**Every alert says how to turn alerts off.** A module with no obvious way
out is not finished (Steven's recording guidance on modules, 2026-09-25).
Here the way out is ``ZENDAVOX_DRIFT=off`` in the environment, or
``"drift": "off"`` in ``~/.zendavox/dev.json``; ``warn`` keeps the alerts
but never stops a message.

Stdlib only, like the rest of this package, and a failure anywhere means
saying nothing rather than costing a message.
"""

from __future__ import annotations

import json
import os
import pathlib
import shutil
import subprocess
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace

from zendavox.dev import drift
from zendavox.dev.config import home_settings

#: Where the setting is read from: the environment first, then the
#: machine-wide settings file under this key.
MODE_ENV = "ZENDAVOX_DRIFT"
MODE_KEY = "drift"

#: Alerts and the stop.
ON = "on"
#: Alerts only; a message is never held back.
WARN = "warn"
#: Nothing at all.
OFF = "off"
MODES = (ON, WARN, OFF)

#: The measurement's settings for live use. Set on 2026-09-26 against 82
#: real chats on this machine: at these, 18 would have seen a notice at
#: some point, 9 a warning, and 4 reached the model check for a stop.
THRESHOLD = 0.15
#: Turns shorter than this answer the assistant's last reply, and are
#: scored against it as well as against the chat.
REPLY_BELOW = 6

#: Messages in a row away from the subject before the quiet line.
NOTICE_RUN = 3
#: ...before the warning. The measurement's own "drifted" point.
WARNING_RUN = drift.RUN_TO_DRIFT
#: ...before the message is held back, if a model agrees.
STOP_RUN = 7

#: A message starting with this is sent whatever the count, and the chat
#: takes the new subject as its own from that message on.
CONTINUE_PREFIX = "continue:"

QUIET = "quiet"
NOTICE = "notice"
WARNING = "warning"
STOP = "stop"

#: How every alert ends: the way to turn alerts off.
OFF_SWITCH = (
    f"To turn these alerts off, set {MODE_ENV}=off, or put "
    f'"{MODE_KEY}": "off" in ~/.zendavox/dev.json.'
)

#: Asked of the model before a stop. One word back, so nothing to parse.
QUESTION = (
    "You are checking whether a chat has left the subject it started on.\n\n"
    "How the chat opened:\n{opening}\n\n"
    "The most recent messages:\n{recent}\n\n"
    "The message about to be sent:\n{prompt}\n\n"
    "Following a problem deeper, or a side step that serves the original "
    "goal, is still ON the subject. A different subject that belongs in a "
    "separate chat is OFF. Answer with exactly one word: ON or OFF."
)

#: The model asked. Small and fast: this runs while the person waits.
MODEL = "claude-haiku-4-5-20251001"
#: Seconds allowed for the answer. The hook's own limit is set above this.
MODEL_TIMEOUT = 20

#: A model check: given the chat's turns and the new message, ``True`` if
#: it has left its subject, ``False`` if not, ``None`` if nobody could say.
Check = Callable[[Sequence[drift.Turn], str], "bool | None"]


@dataclass(frozen=True)
class Verdict:
    """What to do about one message about to be sent."""

    level: str
    #: Messages in a row away from the subject, this one included.
    run: int = 0
    #: The chat's subject, as its first message put it.
    subject: str = ""
    #: What the person sees. ``None`` when there is nothing to say.
    message: str | None = None
    #: What the assistant is told. Only ever set on a warning.
    context: str | None = None


def mode(
    env: Mapping[str, str] | None = None,
    home: Mapping[str, str] | None = None,
) -> str:
    """Which of :data:`MODES` is in force. Anything unrecognised is ``on``.

    Unrecognised reads as on rather than off because a typo in the setting
    should not quietly switch off a feature somebody asked for; the alerts
    themselves say how to switch it off properly.
    """
    environment = os.environ if env is None else env
    settings = home_settings() if home is None else home
    for raw in (environment.get(MODE_ENV), settings.get(MODE_KEY)):
        if raw is None or not raw.strip():
            continue
        value = raw.strip().lower()
        return value if value in MODES else ON
    return ON


def _continues(text: str) -> bool:
    return text.strip().lower().startswith(CONTINUE_PREFIX)


def _from_last_continue(turns: Sequence[drift.Turn]) -> list[drift.Turn]:
    """The turns since the person last said to carry on with a new subject.

    Worked out from the transcript every time rather than kept anywhere, so
    there is no state to go stale: the ``continue:`` message is in the chat,
    and from it on the chat is about what that message was about.
    """
    start = 0
    for position, turn in enumerate(turns):
        if _continues(turn.text):
            start = position
    kept = list(turns[start:])
    if kept and _continues(kept[0].text):
        text = kept[0].text.strip()[len(CONTINUE_PREFIX) :].strip()
        kept[0] = replace(kept[0], text=text)
    return [replace(turn, index=position) for position, turn in enumerate(kept)]


def assess(
    turns: Sequence[drift.Turn],
    prompt: str,
    *,
    reply: str = "",
    setting: str = ON,
    check: Check | None = None,
    path: pathlib.Path | None = None,
) -> Verdict:
    """Decide what happens to ``prompt``, given the chat so far.

    ``reply`` is what the assistant said since the last turn; ``check`` is
    asked before any stop, and defaults to :func:`ask_model`.
    """
    if setting == OFF:
        return Verdict(QUIET)
    if _continues(prompt):
        # The person has read the stop and chosen. Never argue with that.
        return Verdict(QUIET)
    if not drift.tokens(prompt) or drift.is_housekeeping(prompt):
        # "yes", "go ahead", "run the checks": not a change of subject.
        return Verdict(QUIET)

    everything = [
        *turns,
        drift.Turn(index=len(turns), at=None, text=prompt, reply=reply),
    ]
    window = _from_last_continue(everything)
    reading = drift.score(
        window,
        path=path or pathlib.Path("prompt"),
        threshold=THRESHOLD,
        hold_away=True,
        reply_below=REPLY_BELOW,
    )
    if reading.state == drift.UNMEASURABLE or not reading.points:
        return Verdict(QUIET)

    run = reading.points[-1].run
    # What the person is told: the messages actually away, counted back
    # from this one. ``run`` counts the smoothed score, which trails the
    # messages by up to a window, so it decides the level but would
    # under-state the count on screen.
    away = _away_in_a_row(reading.points)
    subject = window[0].opening
    if run >= STOP_RUN and setting == ON:
        judged = (check or ask_model)(window[:-1], prompt)
        if judged is False:
            # The model read it and says this is still the same subject.
            # Believe it: the count is the cruder of the two readings.
            return Verdict(QUIET, run=run, subject=subject)
        if judged is True:
            return Verdict(
                STOP,
                run=run,
                subject=subject,
                message=(
                    f'Stopped: this chat started on "{subject}", and this '
                    f"is message {away} in a row about something else, so it "
                    "was not sent. Start a new chat for the new subject. To "
                    "carry on here anyway, send it again starting with "
                    f'"{CONTINUE_PREFIX}" and that becomes this chat\'s '
                    "subject from then on. " + OFF_SWITCH
                ),
            )
        # Nobody could confirm it: warn, never stop on the count alone.
    if run >= WARNING_RUN:
        left = STOP_RUN - run
        countdown = (
            f"It will stop after {left} more message{'s' if left != 1 else ''} "
            "like this. "
            if setting == ON and left > 0
            else ""
        )
        return Verdict(
            WARNING,
            run=run,
            subject=subject,
            message=(
                f'Off topic: this chat started on "{subject}", and the last '
                f"{away} messages have been about something else. "
                f"{countdown}Coming back to the subject clears this; for a "
                "new subject, start a new chat or begin your message with "
                f'"{CONTINUE_PREFIX}". ' + OFF_SWITCH
            ),
            context=(
                f"Zendavox topic check: the person's last {away} messages "
                f'have moved away from what this chat started on ("{subject}"). '
                "Answer this message as usual, then offer once, in one "
                "sentence, to take the new subject to a new chat."
            ),
        )
    if run >= NOTICE_RUN:
        return Verdict(
            NOTICE,
            run=run,
            subject=subject,
            message=(
                f"Topic check: the last {away} messages have moved away from "
                f'what this chat started on ("{subject}"). ' + OFF_SWITCH
            ),
        )
    return Verdict(QUIET, run=run, subject=subject)


def _away_in_a_row(points: Sequence[drift.Point]) -> int:
    """Scored messages below the line, counted back from the last one."""
    count = 0
    for point in reversed(points):
        if point.continuity is None:
            continue  # "yes", housekeeping: neither away nor back
        if point.continuity >= THRESHOLD:
            break
        count += 1
    return count


def hook_output(verdict: Verdict) -> str:
    """The verdict as the harness reads it from a prompt hook's stdout.

    An empty string lets the message through with nothing said.
    """
    if verdict.level == STOP:
        return json.dumps({"decision": "block", "reason": verdict.message})
    if verdict.message is None:
        return ""
    output: dict[str, object] = {"systemMessage": verdict.message}
    if verdict.context:
        output["hookSpecificOutput"] = {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": verdict.context,
        }
    return json.dumps(output)


# --------------------------------------------------------------------------
# the second opinion
# --------------------------------------------------------------------------
def question(turns: Sequence[drift.Turn], prompt: str) -> str:
    """What the model is asked: the opening, the latest turns, the message."""

    def lines(chosen: Sequence[drift.Turn], width: int) -> str:
        return (
            "\n".join(f"- {' '.join(t.text.split())[:width]}" for t in chosen)
            or "- (none)"
        )

    return QUESTION.format(
        opening=lines(turns[: drift.WARMUP_TURNS], 400),
        recent=lines(turns[drift.WARMUP_TURNS :][-6:], 300),
        prompt=" ".join(prompt.split())[:600],
    )


def _answer(text: str) -> bool | None:
    words = text.strip().upper().split()
    if not words:
        return None
    first = words[0].strip(".,!\"'")
    return {"OFF": True, "ON": False}.get(first)


def claude_executable(env: Mapping[str, str] | None = None) -> str | None:
    """The Claude Code program on this machine, if there is one.

    Named in ``ZENDAVOX_DRIFT_CLAUDE``, else on the PATH, else the newest
    copy the desktop app keeps under ``%APPDATA%/Claude/claude-code``.
    """
    environment = os.environ if env is None else env
    named = environment.get("ZENDAVOX_DRIFT_CLAUDE")
    if named and pathlib.Path(named).is_file():
        return named
    found = shutil.which("claude")
    if found:
        return found
    appdata = environment.get("APPDATA")
    if appdata:
        copies = sorted(
            pathlib.Path(appdata, "Claude", "claude-code").glob("*/claude.exe"),
            key=lambda p: p.stat().st_mtime,
        )
        if copies:
            return str(copies[-1])
    return None


def _ask_claude_code(text: str) -> bool | None:
    program = claude_executable()
    if program is None:
        return None
    # The child is itself a Claude Code session, and would run this very
    # hook on its own prompt. Switched off for it, and started with no
    # project or user settings so no other hook runs either.
    environment = {**os.environ, MODE_ENV: OFF}
    try:
        done = subprocess.run(
            [
                program,
                "-p",
                "--model",
                "haiku",
                "--setting-sources",
                "",
                "--no-session-persistence",
            ],
            input=text,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=MODEL_TIMEOUT,
            env=environment,
            cwd=pathlib.Path.home(),
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if done.returncode != 0:
        return None
    return _answer(done.stdout)


def _ask_api(text: str) -> bool | None:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return None
    request = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=json.dumps(
            {
                "model": MODEL,
                "max_tokens": 5,
                "messages": [{"role": "user", "content": text}],
            }
        ).encode("utf-8"),
        headers={
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=MODEL_TIMEOUT) as reply:
            body = json.loads(reply.read().decode("utf-8"))
    except (OSError, ValueError):
        return None
    parts = body.get("content") if isinstance(body, dict) else None
    if not isinstance(parts, list):
        return None
    said = " ".join(str(p.get("text", "")) for p in parts if isinstance(p, dict))
    return _answer(said)


def ask_model(turns: Sequence[drift.Turn], prompt: str) -> bool | None:
    """Has this chat really left its subject? ``None`` if nobody could say.

    Claude Code first, which costs nothing extra where it has been signed
    in on its own (the desktop app's sign-in does not carry over to it: on
    Steven's machine on 2026-09-26 it answered "Not logged in"); the API
    second, for a machine with a key.
    """
    text = question(turns, prompt)
    for ask in (_ask_claude_code, _ask_api):
        judged = ask(text)
        if judged is not None:
            return judged
    return None
