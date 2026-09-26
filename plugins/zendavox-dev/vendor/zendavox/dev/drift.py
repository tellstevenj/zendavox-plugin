"""Topic drift inside one chat: the local measurement (P0).

A chat starts on one thing and, some way in, is about another. Nobody
decided that; it happened a turn at a time. This module measures it from
the transcript the app already keeps on disk, so the thresholds that will
later run inside the product are set against chats that really happened
rather than guessed at a whiteboard.

**This is not the drift that** :mod:`tests.test_dev_contract` **guards.**
That one is two sides of a protocol growing apart. This one is a
conversation wandering off its own subject, which is what Steven asked for
on 2026-09-18.

**Why it lives here rather than in the connector.** The connector never
sees a conversation - :mod:`zendavox.connector` keeps it that way on
purpose, so the record is correct whichever assistant is on the other end.
It can only infer a topic from what a session writes. The local transcript
has the turns themselves, so this is the only place the inference can be
checked against the thing it is inferring. Whatever the product ends up
shipping, this is where its numbers come from.

**What it reads.** :mod:`zendavox.dev.grouping` already locates every
thread: ``~/.claude/projects/<cwd, flattened>/<cliSessionId>.jsonl``, one
JSON object per line. Only the person's own typed turns are scored. A
transcript is read and never written, and nothing here reaches the
network or a database.

**How it decides.** The first few human turns become the *anchor*: the
vocabulary the chat opened on. Every turn after that is scored on how much
of what was said was already in the anchor, and the score is smoothed over
a short window so one stray sentence is not an event.

The part that matters, and the part these constants exist to tune: a
falling score is not by itself drift. A chat that starts on "the invoice
email is failing" and moves to "the mail layer wants rewriting" has
followed the problem down, not lost it. So a dip is only called drift once
it has run for several turns **without coming back**. A score that
oscillates around the anchor is depth; one that walks away and stays away
is a different subject.

Every number below is a first guess, named so it can be argued with and
overridden from the command line. Setting them is the whole point of
running this.
"""

from __future__ import annotations

import json
import pathlib
import re
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime

#: Human turns used to build the anchor. Three rather than one because an
#: opening line is often "look at X" with the actual subject arriving in the
#: reply to the first question back.
WARMUP_TURNS = 3

#: An anchor smaller than this cannot be scored against - a three-word
#: opening turn makes every later turn look like a departure. Such a chat is
#: reported as unmeasurable rather than as on topic, because "we could not
#: tell" and "it stayed put" are different answers.
MIN_ANCHOR_TERMS = 8

#: Turns the score is averaged over. One turn is noise; three is a drift.
WINDOW = 3

#: Smoothed score below which a turn counts as away from the anchor.
OFF_THRESHOLD = 0.25

#: Consecutive away turns, with no return above the threshold, before the
#: chat is called drifted rather than merely deep.
RUN_TO_DRIFT = 4

#: Words carrying no subject. Deliberately the same shape as the set in
#: :mod:`zendavox.notice`, which the server side already uses to decide
#: whether a record fits its session: if the two disagree about what a word
#: is, the local reading and the shipped feature disagree about everything
#: downstream of it. Kept as a copy rather than an import because this
#: package may not import anything that needs installing, and that module
#: pulls in SQLAlchemy.
STOPWORDS = frozenset(
    {
        "the", "for", "and", "a", "an", "of", "to", "in", "on", "at",
        "re", "fw", "fwd", "record", "records", "request", "requests",
        "pdf", "redacted", "signed",
        # Added here, not in notice.py: these are frequent in a chat and
        # say nothing about its subject. They are scored as absent from
        # every anchor, which is the same as not being there at all.
        "you", "can", "get", "got", "let", "are", "was", "were", "this",
        "that", "with", "have", "has", "had", "not", "but", "its",
        "what", "when", "which", "there", "then", "than", "them", "they",
        "would", "could", "should", "please", "thanks", "yes", "okay",
        "now", "just", "like", "want", "need", "make", "made", "does",
        "did", "done", "going", "know", "see", "look", "say", "said",
        "one", "two", "all", "any", "how", "why", "who", "our", "your",
        "from", "into", "out", "off", "about", "also", "more", "most",
        "some", "each", "every", "here", "been", "being", "will",
    }
)

#: Text that arrives on a ``user`` line without a person having typed it:
#: tool output played back, harness notices, slash-command envelopes. Scoring
#: these would measure the harness rather than the conversation.
INJECTED_PREFIXES = (
    "<task-notification>",
    "<system-reminder>",
    "<ci-monitor-event>",
    "<local-command-",
    "<command-name>",
    "<command-message>",
    "<bash-input>",
    "<bash-stdout>",
    "<bash-stderr>",
    "<user-prompt-submit-hook>",
    "caveat: the messages below",
    "[request interrupted",
    "api error",
    # Found on 2026-09-19 by reading real transcripts: a Stop hook's output
    # arrives on a user line like anything else, and it is long, technical
    # and identical every time. Left in, it reads as the person changing the
    # subject at the end of every chat that has such a hook configured.
    "stop hook feedback:",
    "sessionstart hook",
    "posttooluse:",
    "pretooluse:",
)

#: Turns that are about running the session rather than about its subject:
#: filing the chat, telling it to record, asking for the checks to be run.
#: Found on 2026-09-19 - they appear near the end of almost every chat, they
#: share no vocabulary with anything before them, and left in they fire drift
#: on the way out of a conversation that never changed subject.
#:
#: This matters more for the product than for this module. On the connector
#: side the only evidence of a turn is what it made the session *write*, and
#: housekeeping turns are precisely the ones that write - so the write stream
#: is more crowded with them than the transcript is, not less.
HOUSEKEEPING = (
    "which group should this thread belong to",
    "just name the group",
    "update zendavox",
    # Steven types this one as often as not, and a phrase list that only
    # matches the correct spelling misses the turn entirely. Which is the
    # standing weakness of matching on phrases: it catches what somebody
    # thought to write down. The shipped version wants something that keys
    # off what the turn *does* - a turn that only asks the session to file
    # itself - rather than off how it was worded.
    "update zenavox",
    "anything in this thread that has not yet been recorded",
    "update the record",
    "record this",
    "record what",
    "sweep this",
    "close the session",
    "run the checks",
    "commit and push",
    "what is the next step",
    "do not overwrite any current decisions",
)

#: The states a turn can be in. Named rather than numbered so a printed line
#: says what happened without a key beside it.
WARMING = "warming"
ON_TOPIC = "on_topic"
DESCENDING = "descending"
DRIFTED = "drifted"
UNMEASURABLE = "unmeasurable"


@dataclass(frozen=True)
class Turn:
    """One thing the person typed, in the order they typed it."""

    index: int
    at: datetime | None
    text: str
    #: What the assistant said just before this turn. A short turn ("Save
    #: both", "the second one") answers that, not the chat's opening, and
    #: scored without it reads as the person changing the subject.
    reply: str = ""

    @property
    def opening(self) -> str:
        """The first line, flattened, for printing beside a score."""
        flat = " ".join(self.text.split())
        return flat[:72]


@dataclass(frozen=True)
class Point:
    """One turn, scored two ways.

    Both are kept because the first one was tried on real transcripts on
    2026-09-19 and was wrong, and the comparison is the evidence for that -
    see :data:`OFF_THRESHOLD`.
    """

    turn: Turn
    #: How much of this turn was already in the opening anchor, 0.0 to 1.0.
    #: ``None`` for a turn with no scorable words in it at all ("yes", "go
    #: ahead") - which is not a score of zero, and must not be averaged as
    #: one. Kept for comparison only; it does not decide anything.
    anchor_score: float | None
    #: How much of this turn was already somewhere in the chat before it.
    #: This is the measure the state is taken from.
    continuity: float | None
    #: :attr:`continuity` averaged over the last :data:`WINDOW` scorable
    #: turns.
    smoothed: float | None
    state: str
    #: How many turns in a row have been away from the chat at this point.
    run: int


@dataclass(frozen=True)
class Reading:
    """What one transcript turned out to be."""

    path: pathlib.Path
    session_id: str
    turns: int
    anchor: frozenset[str]
    points: tuple[Point, ...]
    #: Where the chat stood at its last scorable turn. Not sticky: a chat
    #: that drifted and came back ends ``on_topic``, and ``drifted_at`` is
    #: what says it went away at all.
    state: str
    #: The turn the chat was first called drifted at, if it ever was.
    drifted_at: int | None
    #: How many times it came back after being called drifted. This is the
    #: number that separates a chat following a problem downwards from one
    #: that changed subject: the first oscillates, the second does not.
    returns: int = 0
    #: Turns left out of the scoring as session housekeeping rather than
    #: subject. Counted rather than dropped quietly, so a reading that rests
    #: on throwing half the chat away says so.
    housekeeping: int = 0
    #: Why it could not be scored, when ``state`` is ``unmeasurable``.
    note: str | None = None


# --------------------------------------------------------------------------
# reading a transcript
# --------------------------------------------------------------------------
def tokens(value: str) -> set[str]:
    """The subject-bearing words in a piece of text.

    Same rule as ``zendavox.notice._tokens`` - lowercase, three characters
    or more, stopwords dropped - plus this module's wider stopword set. See
    :data:`STOPWORDS` for why the two are kept in step.
    """
    parts = re.split(r"[^a-z0-9]+", value.lower())
    return {p for p in parts if len(p) >= 3 and p not in STOPWORDS}


def _when(raw: object) -> datetime | None:
    if not isinstance(raw, str) or not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None


def _typed_text(entry: dict[str, object]) -> str | None:
    """The text of a turn the person actually typed, or ``None``.

    A ``user`` line in the transcript is not the same thing as a person
    speaking. Tool results come back on one, so do harness notices and
    slash-command envelopes, and every one of them would be scored as the
    person changing the subject.
    """
    if entry.get("type") != "user" or "toolUseResult" in entry:
        return None
    message = entry.get("message")
    if not isinstance(message, dict):
        return None
    content = message.get("content")
    if not isinstance(content, str):
        return None
    text = content.strip()
    if not text:
        return None
    lowered = text.lower()
    if any(lowered.startswith(prefix) for prefix in INJECTED_PREFIXES):
        return None
    return text


def _assistant_text(entry: dict[str, object]) -> list[str]:
    """The prose of an assistant line: its text blocks, not its tool calls."""
    message = entry.get("message")
    if not isinstance(message, dict):
        return []
    content = message.get("content")
    if not isinstance(content, list):
        return []
    return [
        str(block.get("text", ""))
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    ]


def human_turns(path: pathlib.Path) -> tuple[Turn, ...]:
    """Every turn the person typed in one transcript, in order."""
    return conversation(path)[0]


def conversation(path: pathlib.Path) -> tuple[tuple[Turn, ...], str]:
    """Every typed turn, and what the assistant has said since the last one.

    The second half is what the next message will be answering, which the
    prompt hook needs and a finished transcript does not.

    A malformed line is skipped rather than raised on: these files are
    written by a live app, and the last line of a chat still in progress is
    routinely half-written.
    """
    found: list[Turn] = []
    said: list[str] = []
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return (), ""
    for line in raw.splitlines():
        stripped = line.strip()
        # Cheap test before the parse. The prompt hook reads the whole
        # transcript before every message, and most of a long chat is tool
        # output on lines that can never be a typed turn.
        if not stripped or (
            '"user"' not in stripped and '"assistant"' not in stripped
        ):
            continue
        try:
            loaded = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if not isinstance(loaded, dict):
            continue
        entry: dict[str, object] = loaded
        if entry.get("type") == "assistant":
            said.extend(_assistant_text(entry))
            continue
        text = _typed_text(entry)
        if text is None:
            continue
        found.append(
            Turn(
                index=len(found),
                at=_when(entry.get("timestamp")),
                text=text,
                reply=" ".join(said),
            )
        )
        said = []
    return tuple(found), " ".join(said)


def transcripts(root: pathlib.Path) -> Iterator[pathlib.Path]:
    """Every thread under a projects directory, newest first."""
    try:
        folders = sorted(d for d in root.iterdir() if d.is_dir())
    except OSError:
        return
    found: list[pathlib.Path] = []
    for folder in folders:
        try:
            found.extend(folder.glob("*.jsonl"))
        except OSError:
            continue
    yield from sorted(found, key=lambda p: p.stat().st_mtime, reverse=True)


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------
def anchor_of(
    turns: Sequence[Turn], *, warmup: int = WARMUP_TURNS
) -> frozenset[str]:
    """The vocabulary the chat opened on."""
    words: set[str] = set()
    for turn in turns[:warmup]:
        words |= tokens(turn.text)
    return frozenset(words)


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def is_housekeeping(text: str) -> bool:
    """Is this turn about running the session rather than about its subject?

    Matched on the opening of the turn as well as anywhere inside it,
    because these are short instructions that usually stand alone.
    """
    lowered = " ".join(text.lower().split())
    return any(phrase in lowered for phrase in HOUSEKEEPING)


def read(
    path: pathlib.Path,
    *,
    warmup: int = WARMUP_TURNS,
    window: int = WINDOW,
    threshold: float = OFF_THRESHOLD,
    run_to_drift: int = RUN_TO_DRIFT,
    min_anchor: int = MIN_ANCHOR_TERMS,
    hold_away: bool = False,
    min_words: int = 1,
    reply_below: int = 0,
) -> Reading:
    """Score one transcript from end to end."""
    return score(
        human_turns(path),
        path=path,
        warmup=warmup,
        window=window,
        threshold=threshold,
        run_to_drift=run_to_drift,
        min_anchor=min_anchor,
        hold_away=hold_away,
        min_words=min_words,
        reply_below=reply_below,
    )


def score(
    turns: Sequence[Turn],
    *,
    path: pathlib.Path,
    warmup: int = WARMUP_TURNS,
    window: int = WINDOW,
    threshold: float = OFF_THRESHOLD,
    run_to_drift: int = RUN_TO_DRIFT,
    min_anchor: int = MIN_ANCHOR_TERMS,
    hold_away: bool = False,
    min_words: int = 1,
    reply_below: int = 0,
) -> Reading:
    """Score turns already in hand.

    Split from :func:`read` so the prompt hook can score a chat with the
    message not yet sent added to the end - that message is in no
    transcript until the hook has let it through.

    ``hold_away`` keeps a turn that did not connect to the chat out of the
    chat's vocabulary until the conversation comes back. Without it, a new
    subject that is consistent with itself stops looking new after two or
    three turns: its second message shares words with its first, which is
    by then part of "everything said so far", and the run resets. That is
    right for a report on whether a chat came back, and wrong for anything
    that has to stop a chat which never does. Found 2026-09-26, building
    the alerts on top of this.

    ``reply_below`` scores a turn of fewer words than this against the
    assistant's reply before it as well as against the chat. Short turns
    answer a question; long ones say something of their own. Counting the
    reply for every turn was tried on 2026-09-26 and flagged almost
    nothing, because the assistant follows the person into a new subject
    and its reply then vouches for the move.
    """
    anchor = anchor_of(turns, warmup=warmup)
    session_id = path.stem

    def unmeasurable(note: str) -> Reading:
        return Reading(
            path=path,
            session_id=session_id,
            turns=len(turns),
            anchor=anchor,
            points=(),
            state=UNMEASURABLE,
            drifted_at=None,
            note=note,
        )

    if len(turns) <= warmup:
        return unmeasurable(
            f"only {len(turns)} typed turns; nothing after the warm-up"
        )
    if len(anchor) < min_anchor:
        return unmeasurable(
            f"anchor is {len(anchor)} words, under the {min_anchor} needed"
        )

    points: list[Point] = []
    recent: list[float] = []
    run = 0
    drifted_at: int | None = None
    returns = 0
    housekeeping = 0
    state = ON_TOPIC
    #: Everything said so far. Drift is a turn disconnecting from the whole
    #: conversation behind it, not from its opening three lines: a chat that
    #: descends into detail keeps sharing words with its recent history even
    #: as it leaves the anchor behind, and scoring against the anchor alone
    #: called every such chat drifted.
    seen: set[str] = set()
    #: With ``hold_away``: the words of turns that did not connect, waiting
    #: to join :data:`seen` if the chat comes back.
    away: set[str] = set()

    for turn in turns:
        words = tokens(turn.text)
        if turn.index < warmup:
            seen |= words
            points.append(
                Point(
                    turn=turn,
                    anchor_score=None,
                    continuity=None,
                    smoothed=None,
                    state=WARMING,
                    run=0,
                )
            )
            continue
        if len(words) < min_words or is_housekeeping(turn.text):
            # "yes", "go ahead", "do it" - the person agreeing, not changing
            # the subject; or a housekeeping turn, which is about the session
            # rather than its subject. Carried at whatever state it was
            # already in, and left out of the average either way.
            if len(words) >= min_words:
                housekeeping += 1
            points.append(
                Point(
                    turn=turn,
                    anchor_score=None,
                    continuity=None,
                    smoothed=_mean(recent),
                    state=state,
                    run=run,
                )
            )
            continue
        anchor_score = len(words & anchor) / len(words)
        context = (
            seen | tokens(turn.reply) if len(words) < reply_below else seen
        )
        continuity = len(words & context) / len(words)
        if not hold_away:
            seen |= words
        elif continuity >= threshold:
            # Connected. Whatever was said while away is part of the chat
            # now: it came back, so the aside was on the way somewhere.
            seen |= words | away
            away.clear()
        else:
            away |= words
        recent.append(continuity)
        del recent[:-window]
        smoothed = _mean(recent)
        if smoothed is None:  # pragma: no cover - recent just took a value
            smoothed = continuity
        if smoothed < threshold:
            run += 1
        else:
            run = 0
        if run >= run_to_drift:
            if state != DRIFTED:
                state = DRIFTED
                if drifted_at is None:
                    drifted_at = turn.index
        elif run > 0:
            state = DESCENDING
        else:
            # Back above the line. The state says where the chat is now, not
            # where it has ever been - ``drifted_at`` keeps the history, and
            # a return is counted because coming back is the whole signal
            # that this was depth rather than a change of subject.
            if state == DRIFTED:
                returns += 1
            state = ON_TOPIC
        points.append(
            Point(
                turn=turn,
                anchor_score=anchor_score,
                continuity=continuity,
                smoothed=smoothed,
                state=state,
                run=run,
            )
        )

    return Reading(
        path=path,
        session_id=session_id,
        turns=len(turns),
        anchor=anchor,
        points=tuple(points),
        state=state,
        drifted_at=drifted_at,
        returns=returns,
        housekeeping=housekeeping,
    )


# --------------------------------------------------------------------------
# saying what was found
# --------------------------------------------------------------------------
def describe(reading: Reading, *, verbose: bool = False) -> str:
    """One chat, in words, for a person reading the output."""
    lines: list[str] = [f"{reading.session_id}  {reading.turns} typed turns"]
    if reading.state == UNMEASURABLE:
        lines.append(f"  unmeasurable - {reading.note}")
        return "\n".join(lines)
    # A chat can have drifted and come back, so the two are reported
    # separately rather than collapsed into one word: "ended on topic, first
    # drifted at turn 41" is a different chat from "still drifted".
    since = (
        f", first drifted at turn {reading.drifted_at}"
        if reading.drifted_at is not None
        else ""
    )
    extra = ""
    if reading.returns:
        extra += f", came back {reading.returns}x"
    if reading.housekeeping:
        extra += f", {reading.housekeeping} housekeeping turns set aside"
    lines.append(
        f"  anchor {len(reading.anchor)} words  ->  "
        f"ended {reading.state}{since}{extra}"
    )
    if verbose:
        lines.append("      turn  anch  cont  smth")
        for point in reading.points:
            if point.state == WARMING:
                mark = "  ."
            elif point.state == DRIFTED:
                mark = "  X"
            elif point.state == DESCENDING:
                mark = "  ~"
            else:
                mark = "   "
            lines.append(
                f"{mark} {point.turn.index:>5} "
                f"{_figure(point.anchor_score)} "
                f"{_figure(point.continuity)} "
                f"{_figure(point.smoothed)}  {point.turn.opening}"
            )
    return "\n".join(lines)


def _figure(value: float | None) -> str:
    return "   -" if value is None else f"{value:4.2f}"


def summarise(readings: Iterable[Reading]) -> str:
    """The counts across every chat looked at."""
    counts: dict[str, int] = {}
    total = 0
    for reading in readings:
        total += 1
        counts[reading.state] = counts.get(reading.state, 0) + 1
    if not total:
        return "No transcripts found."
    order = (DRIFTED, DESCENDING, ON_TOPIC, UNMEASURABLE)
    parts = [f"  {name:<14} {counts.get(name, 0)}" for name in order]
    return "\n".join([f"{total} chats read", *parts])
