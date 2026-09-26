"""Reminders to record, raised by what the person just typed.

Two kinds of thing were not being recorded, and for the same reason: they
turn up in the middle of other work, and recording them depended on an
assistant noticing. Steven, 2026-09-26, after the recording guidance of
2026-09-25 had been in place a day and a half and neither kind had a single
entry:

**Parked items.** "Put it on the to-do list", "come back to it later" - said
in passing, and gone when the chat ends. His guidance: record what is still
to do, what was already found, and why it was parked.

**Service notices.** Supabase, Render, GitHub, Stripe, Google and the rest
announce changes by email; when one is pasted into a chat it should be
recorded with what it says, the date it takes effect, and whether it
affects Zendavox.

So the prompt hook reads each message for the signs of either and, when it
finds them, tells the assistant - on that message, not at the end - what to
record and how. It can be wrong in both directions: "later" is an ordinary
word. A reminder that did not apply costs the assistant one sentence of
reading; a parked item missed is lost. The wording says "if", and the
assistant decides.

Entries are written so they can be found again: a parked item as an
activity starting ``PARKED:``, a notice as a finding starting
``SERVICE NOTICE:``. The stuck-work sweep looks for the first; the daily
mail check writes the second.

``ZENDAVOX_REMINDERS=off`` (or ``"reminders": "off"`` in
``~/.zendavox/dev.json``) switches this off.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping

from zendavox.dev.config import home_settings

MODE_ENV = "ZENDAVOX_REMINDERS"
MODE_KEY = "reminders"

#: How a person parks something. Matched as whole phrases, lowercase.
PARKING = re.compile(
    r"\b("
    r"to-?do list|on the to-?do|put (it|that|this) on the (list|backlog)"
    r"|(add|put) (it|that|this) (to|on) the (list|backlog)"
    r"|come back to (it|this|that)|circle back|revisit (it|this|that)"
    r"|(do|handle|deal with|pick) (it|this|that)( up)? (later|tomorrow)"
    r"|park (it|this|that)|parked|park for now|table (it|this) for"
    r"|leave (it|this|that) for (now|later)|not now|some other time"
    r"|later on|for later|tomorrow"
    r")\b"
)

#: Who sends the notices that matter here.
SERVICES = re.compile(
    r"\b(supabase|render|github|stripe|google|gmail|resend|anthropic|claude"
    r"|cloudflare|namecheap|godaddy|zapier|openai|python|postgres)\b"
)

#: What a change notice says. One of these beside a service's name is the
#: sign; either alone is ordinary conversation.
NOTICE_WORDS = re.compile(
    r"\b(effective (on|from|as of)|takes effect|will take effect"
    r"|(will be|is being|are) deprecat\w*|end of life|end-of-life"
    r"|will be sunset|sunsetting|will be (removed|retired"
    r"|disabled|discontinued)|breaking change|action required"
    r"|we('| a)re (changing|updating)|changes to (our|your)|price (change|increase)"
    r"|pricing (change|update)|terms of service|privacy policy update"
    r"|scheduled maintenance|migrat(e|ion) (by|before)|no longer (be )?supported)\b"
)

#: A notice names when. Without a date it is talk about a change, not a
#: notice of one - found 2026-09-26, when the first version fired on five
#: pasted tracebacks and a spec, none of them notices.
DATE = re.compile(
    r"\b(\d{4}-\d{2}-\d{2}|(january|february|march|april|may|june|july"
    r"|august|september|october|november|december|jan|feb|mar|apr|jun|jul"
    r"|aug|sep|sept|oct|nov|dec)\.? \d{1,2}(st|nd|rd|th)?)\b"
)

#: Pasted error output is not a notice, whatever it mentions.
TRACEBACK = re.compile(r"traceback \(most recent call last\)|\bfile \"")

#: A notice is usually pasted, so it is long. A short message naming a
#: service and saying "deprecated" is usually talk about code.
NOTICE_MIN_CHARS = 300

PARKED_REMINDER = (
    "Zendavox reminder: this message may be parking something for later. "
    "If it is, record it now with the Zendavox note_activity tool, kind "
    '"note", as a description starting "PARKED:" that says what is still to '
    "do, what was already found, and why it was parked - so it survives this "
    "chat. If nothing is being parked, ignore this."
)

NOTICE_REMINDER = (
    "Zendavox reminder: this message looks like a notice of a change from a "
    "service Zendavox uses. If it is, record it now with the Zendavox "
    'note_activity tool, kind "finding", as a description starting "SERVICE '
    'NOTICE:" that gives who sent it, what changes, the date it takes '
    "effect, whether it affects Zendavox, and why or why not. If it is not a "
    "notice, ignore this."
)


def enabled(
    env: Mapping[str, str] | None = None,
    home: Mapping[str, str] | None = None,
) -> bool:
    environment = os.environ if env is None else env
    settings = home_settings() if home is None else home
    for raw in (environment.get(MODE_ENV), settings.get(MODE_KEY)):
        if raw is not None and raw.strip():
            return raw.strip().lower() != "off"
    return True


def is_parking(text: str) -> bool:
    return PARKING.search(text.lower()) is not None


def is_notice(text: str) -> bool:
    lowered = text.lower()
    return (
        len(text) >= NOTICE_MIN_CHARS
        and SERVICES.search(lowered) is not None
        and NOTICE_WORDS.search(lowered) is not None
        and DATE.search(lowered) is not None
        and TRACEBACK.search(lowered) is None
    )


def for_prompt(text: str) -> str | None:
    """What to tell the assistant about this message, if anything."""
    said: list[str] = []
    if is_notice(text):
        said.append(NOTICE_REMINDER)
    if is_parking(text):
        said.append(PARKED_REMINDER)
    return "\n\n".join(said) if said else None
