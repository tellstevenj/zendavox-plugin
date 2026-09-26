"""Turning a brief into the paragraphs a session reads first.

The connector returns a brief as JSON because JSON is what a program wants.
The thing on the other end of this is a language model reading its opening
context, and handing it raw JSON wastes the one chance to be understood: it
would parse, but it would read as a data dump rather than as what the project
knows.

So this renders the same content as prose and short lists, in the order a
person would want it:

    preferences  - how this person wants to be worked with, first, because it
                   governs everything the session then does
    guidance     - what they want recorded here, and why, so the session
                   knows what to watch for before it starts working
    facts        - what is durably true about the project
    decisions    - what was settled, so it is not reopened by accident
    notices      - what is waiting, counted plainly
    activity     - what happened recently

Two rules carry over from the product. Nothing is silently truncated: where a
list is shortened the text says how many were left out - and, for facts, names
them - because a brief that says "3 open" when 47 are waiting is exactly the
under-reporting Zendavox exists to prevent. And an empty brief says it is empty
rather than rendering nothing, so a first session can tell "this project has no
record yet" apart from "the brief failed to load".

The first of those rules was half-kept until 2026-09-18. Notices were counted
against the server's own total; decisions and activity were not, so a
brief that received the newest ten of each said nothing whatsoever about the
other 35 decisions and 127 activity entries - the totals were in the JSON the
whole time and this file dropped them. Facts were worse: all of them arrived,
twelve rendered, and the server sorted them by key, so the same 26 were absent
from every brief anyone ever read, chosen by alphabet. Counting an omission is
also not enough on its own, because a session cannot ask for what it does not
know exists; omitted facts are now named.
"""

from __future__ import annotations

from typing import Any

#: How many of each list to show. A brief is an orientation, not an archive;
#: the session can ask Zendavox for more once it knows what it is looking for.
#: This is the default, and it suits the short lists - preferences, guidance,
#: notices - where twelve is already more than a project usually holds.
MAX_ITEMS = 12

#: Facts get a far higher ceiling than the short lists, because "what is
#: durably true about this project" is the one section where not knowing an
#: entry exists is the whole failure. Measured 2026-09-18 on this project's
#: own record: 38 facts, 12 rendered, 26 silently absent from every brief
#: ever read - and because the server used to sort them by key, it was the
#: same 26 every time, chosen by alphabet. Facts now arrive newest-first and
#: anything past this ceiling is named below the list rather than counted.
MAX_FACTS = 40

#: Standing decisions are the most expensive thing for a session not to know:
#: the cost of missing one is reopening a question that was settled on
#: purpose. Ten was the old server limit; 45 were standing here when it was
#: measured.
MAX_DECISIONS = 25

#: Recent activity is a trail rather than a record of what is true, so it
#: stays short - but the true total is now stated, which it was not.
MAX_ACTIVITY = 12

#: Long values are almost always pasted prose. Keep the shape of the brief
#: readable and let the session ask for the rest.
MAX_VALUE = 400

#: A source is provenance, not content: it exists so a reader can recognise
#: where something came from and go and look. Recognising it takes a few
#: words, and the full string is a `find_fact` away. Sources here run to 120
#: characters apiece, which across 38 facts was 4,500 characters of the
#: opening context spent on paths and dates.
MAX_SOURCE = 80

#: A rationale gets less room than the decision it explains. What stops a
#: session reopening a settled question is the question and the answer; the
#: reasoning behind it matters when the choice looks odd, and that is the
#: moment to `look_up` the whole thing.
MAX_RATIONALE = 280

#: Recording guidance is not a value to skim - it is the customer's own
#: instruction about what to write down, and half of one is worse than none.
#: It gets a far longer leash than anything else here, and the section below
#: still says plainly when something was cut.
MAX_GUIDANCE = 2000


def _rows(brief: dict[str, Any], key: str) -> list[dict[str, Any]]:
    """Pull a list of records out of a brief, tolerating a changed shape.

    This client and the server it talks to are versioned separately, and a
    server that grows a field must not break a session that has not caught up.
    """
    value = brief.get(key)
    if not isinstance(value, list):
        return []
    return [row for row in value if isinstance(row, dict)]


def _text(row: dict[str, Any], key: str, limit: int = MAX_VALUE) -> str:
    value = row.get(key)
    if value is None:
        return ""
    out = " ".join(str(value).split())
    return out if len(out) <= limit else out[: limit - 1] + "…"


def _instruction(row: dict[str, Any], key: str) -> str:
    """The same, for text that is an instruction rather than a value."""
    value = row.get(key)
    if value is None:
        return ""
    out = " ".join(str(value).split())
    if len(out) <= MAX_GUIDANCE:
        return out
    return out[: MAX_GUIDANCE - 1] + "… (cut here - ask for the rest)"


def _age(row: dict[str, Any]) -> str:
    """" _(first recorded 2026-08-19)_" when a fact is older than its last
    write, and nothing when the two are the same day.

    A fact's age is what tells a reader whether to re-verify it, and
    ``updated_at`` alone cannot carry it: setting a key again moves that
    timestamp even when the value did not change. Only the day is compared,
    and only the day is shown - an ISO timestamp starts with it, which is
    also why this needs no date parsing and no import.
    """
    first = _text(row, "created_at")[:10]
    last = _text(row, "updated_at")[:10]
    if not first or not last or first >= last:
        return ""
    return f" _(first recorded {first})_"


def _undated(row: dict[str, Any]) -> str:
    """" _(no date on record)_" for an entry nobody could date.

    The section this appears in is called "What happened recently", and an
    entry with no date is not recently anything. Saying so on the line is
    what stops a reader inferring a date from the company it keeps: these
    sort by when they were recorded, which is the one thing known about
    them, and that is not a claim about when they happened.
    """
    if row.get("occurred_at") is None and row.get("date_certainty") == 0:
        return " _(no date on record)_"
    if row.get("decided_at") is None and row.get("date_certainty") == 0:
        return " _(no date on record)_"
    return ""


def _count(brief: dict[str, Any], key: str) -> int:
    value = brief.get(key)
    return value if isinstance(value, int) and value >= 0 else 0


def _section(
    lines: list[str],
    title: str,
    items: list[str],
    total: int | None = None,
    *,
    limit: int = MAX_ITEMS,
    names: list[str] | None = None,
    reach_the_rest: str = "",
) -> None:
    """Add a section, and say plainly what was left out.

    ``names`` is the label of every item in ``items``, in the same order. When
    the list is shortened, the ones that did not make it are named rather than
    only counted: a session cannot ask for what it does not know exists, so a
    bare "…and 26 more" leaves those 26 as good as unrecorded. ``total`` is
    the server's own count where it sends one, because a shortened list that
    counts only the rows it received would claim the record is as short as
    the page.
    """
    if not items:
        return
    lines.append(f"### {title}")
    lines.extend(items[:limit])
    shown = min(len(items), limit)
    hidden = (total if total is not None else len(items)) - shown
    if hidden > 0:
        note = f"_…and {hidden} more not shown here."
        left_out = [] if names is None else names[limit:]
        if left_out:
            note += " They are recorded, not missing — " + ", ".join(left_out)
            note += "."
        if reach_the_rest:
            note += f" {reach_the_rest}"
        lines.append(note.rstrip() + "_")
    lines.append("")


def _render_withheld(brief: dict[str, Any], heading: str) -> str:
    """A brief the server held back because this key kept reading without
    writing. Only counts arrive, so only counts are shown - rendering it as
    an ordinary brief would announce "no previous sessions", which is the
    one thing it must not say."""
    lines = [heading, "", f"**{_text(brief, 'writes_owed', 1000)}**", ""]
    counts = brief.get("withheld_counts")
    items: list[str] = []
    if isinstance(counts, dict):
        for name, value in counts.items():
            if isinstance(value, dict):
                items.extend(
                    f"- activity, {kind}: {n}"
                    for kind, n in value.items()
                    if isinstance(n, int)
                )
            elif isinstance(value, int):
                items.append(f"- {name.replace('_', ' ')}: {value}")
    if items:
        lines.append("### Held back until something is recorded")
        lines.extend(items)
        lines.append("")
    tool = _text(brief, "look_up_tool") or "look_up"
    lines.append(
        f"To fetch one specific item on purpose, use `{tool}` with a few "
        "words that name it."
    )
    lines.append("")
    rules = brief.get("rules")
    if isinstance(rules, dict) and isinstance(rules.get("text"), str):
        lines.append("## Operating rules")
        lines.append(rules["text"].strip())
    return "\n".join(lines).rstrip() + "\n"


def render(brief: dict[str, Any], *, project_label: str | None = None) -> str:
    """Render a brief as Markdown for a session's opening context.

    ``project_label`` overrides what the brief itself says, for a caller
    that already knows a better name to show; left unset, this reads the
    project's own name and account straight out of the brief - so any
    assistant asked "which project is this" has an answer without a
    separate lookup. Steven, 2026-09-05, pointing a connector at this
    project and asking it to identify itself: "ask Zendavox what project
    in Zendavox that this project in code is attached to.\""""
    heading = "## Zendavox project brief"
    label = project_label
    if label is None:
        name = _text(brief, "project_name")
        account = _text(brief, "account_name")
        if name and account:
            label = f"{name} ({account} account)"
        elif name:
            label = name
    if label:
        heading = f"{heading} — {label}"

    if brief.get("withheld") is True:
        return _render_withheld(brief, heading)

    sessions = _count(brief, "session_count")
    if sessions == 0:
        opening = (
            "No previous sessions are recorded for this project. "
            "You are the first; what you record now is what the next "
            "session will start from."
        )
    else:
        opening = (
            f"This project has {sessions} recorded "
            f"{'session' if sessions == 1 else 'sessions'} behind it. "
            "What follows is what they established — treat it as already "
            "settled rather than re-deriving it."
        )

    lines = [heading, "", opening, ""]

    _section(
        lines,
        "How this person wants to be worked with",
        [
            f"- **{_text(r, 'key')}**: {_text(r, 'value')}"
            + (
                " _(set for this project)_"
                if _text(r, "scope") == "project"
                else ""
            )
            for r in _rows(brief, "preferences")
        ],
    )
    _section(
        lines,
        "What this project wants recorded, and why",
        [
            f"- **{_text(r, 'topic')}**: {_instruction(r, 'guidance')}"
            + (
                f"\n  _Why: {_instruction(r, 'reason')}_"
                if _text(r, "reason")
                else ""
            )
            + (
                "\n  _(set for this project)_"
                if _text(r, "scope") == "project"
                else ""
            )
            for r in _rows(brief, "recording_guidance")
        ],
    )
    facts = _rows(brief, "facts")
    _section(
        lines,
        "What is true about this project",
        [
            f"- **{_text(r, 'key')}**: {_text(r, 'value')}"
            + (
                f" _(source: {_text(r, 'source', MAX_SOURCE)})_"
                if _text(r, "source")
                else ""
            )
            + (f" _(via {_text(r, 'via')})_" if _text(r, "via") else "")
            + _age(r)
            for r in facts
        ],
        limit=MAX_FACTS,
        # Named, not counted: `find_fact` takes an exact key, so a session
        # that can read the keys can read every one of them.
        names=[_text(r, "key") for r in facts],
        reach_the_rest="Read any of them with `find_fact`.",
    )
    decisions = _rows(brief, "decisions")
    _section(
        lines,
        "What has already been decided",
        [
            f"- {_text(r, 'question')} → **{_text(r, 'decision')}**"
            + _undated(r)
            + (f" _(via {_text(r, 'via')})_" if _text(r, "via") else "")
            + (
                f"\n  _Why: {_text(r, 'rationale', MAX_RATIONALE)}_"
                if _text(r, "rationale")
                else ""
            )
            for r in decisions
        ],
        # The server sends the newest few and the true count separately. Left
        # unused, as it was until 2026-09-18, this section showed 10 of 45
        # standing decisions and said nothing at all about the other 35.
        total=max(_count(brief, "decision_total"), len(decisions)),
        limit=MAX_DECISIONS,
        reach_the_rest=(
            "Search them with `look_up`, or read them all through `/changes`."
        ),
    )

    notices = _rows(brief, "open_notices")
    _section(
        lines,
        "What is open and waiting",
        [
            f"- [{_text(r, 'kind')}] {_text(r, 'message')}"
            + (
                f" _({_text(r, 'disposition')})_"
                if _text(r, "disposition")
                else ""
            )
            for r in notices
        ],
        # The server sends the true total separately, precisely so a shortened
        # list cannot read as the whole picture.
        total=max(_count(brief, "open_notice_total"), len(notices)),
    )
    activity = _rows(brief, "activity")
    _section(
        lines,
        "What happened recently",
        [
            f"- [{_text(r, 'kind')}] {_text(r, 'description')}"
            + _undated(r)
            + (
                f" _(source: {_text(r, 'source', MAX_SOURCE)})_"
                if _text(r, "source")
                else ""
            )
            for r in activity
        ],
        # Same hole as decisions had: the count exists, it just never reached
        # the page. Deploys are left out of this list by the server and are
        # still counted in the total, so the two numbers will not agree - the
        # total is the one to go by.
        total=max(_count(brief, "activity_total"), len(activity)),
        limit=MAX_ACTIVITY,
        reach_the_rest="The rest is in `/changes` and in the session trail.",
    )

    if len(lines) == 4:
        lines.append(
            "_The record is empty so far — no facts, decisions, or notices._"
        )
        lines.append("")

    silence = brief.get("silence")
    if isinstance(silence, dict) and silence.get("silent"):
        reads = _count(silence, "reads_since_write")
        lines.append(
            f"**You have read this brief {reads} times without recording "
            "anything.** Reading is not recording. Record what has happened "
            "since your last entry now, or tell the person plainly that "
            "nothing has been recorded and why."
        )
        lines.append("")

    rules = brief.get("rules")
    rules_text = (
        rules.get("text")
        if isinstance(rules, dict) and isinstance(rules.get("text"), str)
        else None
    )
    if rules_text:
        # The full, current operating rules - not a hand-kept paraphrase of
        # them. rules.py's own docstring already promised this: "fetched
        # fresh at the start of every session... always current." It never
        # actually reached an assistant reading this rendered text until
        # now - every render() call quietly dropped the brief's "rules"
        # field and showed a short, drifting summary instead. Found
        # 2026-09-05 while checking that every connected assistant (not
        # just this one) gets the sweep rule.
        version = rules.get("rules_version") if isinstance(rules, dict) else None
        lines.append(
            f"## Operating rules (version {version})"
            if version is not None
            else "## Operating rules"
        )
        lines.append(rules_text.strip())
        lines.append("")
    else:
        lines.append(
            "Record as you go with the Zendavox tools — `note_decision` when a "
            "question is settled, `note_fact` when something durable is "
            "established, `note_activity` when something is done — and "
            "`close_session` with a summary before you finish. None of this "
            "session carries to the next one unless it is recorded. Record each "
            "change the moment it happens, as its own entry - never hold a "
            "record until the task around it is finished; completion is a "
            "separate entry."
        )
    return "\n".join(lines).strip() + "\n"
