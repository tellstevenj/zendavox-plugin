"""The tools a session records with, spoken over MCP.

The hook in :mod:`zendavox.dev.hooks` covers reading, because a read that has
to be remembered will be forgotten. Writing is the other half, and it cannot be
automated the same way: only the session knows that the thing it just worked
out was a *decision* rather than a passing thought. :mod:`zendavox.connector`
draws the line in the same place - the assistant recognises, Zendavox stores -
and this module is that line made concrete.

Matching the connector's own recording surface, plus a couple of narrow reads
for when the whole brief is more than a session needs:

    brief           - what this project already knows
    open_session    - name the work being done
    note_decision   - a question settled, and why
    note_fact       - something durable about the project
    note_activity   - something done
    find_fact       - look up one fact by key, without re-fetching the brief
    note_preference - how this person wants to be worked with
    note_guidance   - what they want recorded, and why
    close_session   - what the session amounted to

Deleting a recorded fact, decision, or activity is deliberately not a tool
here. It exists in the connector for a customer's own use (the Connect page,
a script), but handing an assistant a one-call way to erase project memory is
a different, larger decision than handing it a way to look one thing up.

**The session record opens lazily.** Nothing is written until the first write
tool is called, and if none ever is, no session record appears. A record full
of empty sessions - one for every terminal opened and closed again - would bury
the real ones, and the point of the record is that it can be skimmed.

**The session id is ours to track, not the model's to pass.** It lives in
:mod:`zendavox.dev.state`, keyed by the harness's session id, so the hooks and
this server agree on it without the model having to thread it through every
call. One less thing to get wrong, and one less thing to hallucinate.

Transport is newline-delimited JSON-RPC 2.0 on stdin and stdout, which is what
MCP's stdio transport is. It is implemented here rather than taken from a
library for the same reason this repository serves its own HTTP: the thing that
has to run on a customer's machine should need a Python interpreter and nothing
else. **Nothing but protocol messages may go to stdout** - every diagnostic in
this module goes to stderr, because one stray print corrupts the stream.
"""

from __future__ import annotations

import json
import pathlib
import sys
from dataclasses import dataclass, field
from typing import Any, TextIO

from zendavox.dev import state
from zendavox.dev.brief_text import render
from zendavox.dev.client import Client, DevError
from zendavox.dev.config import DevConfig, load

SERVER_NAME = "zendavox"

#: Versions of the MCP protocol this server is known to speak. A client asking
#: for one of these is answered in its own version rather than corrected.
KNOWN_VERSIONS = ("2024-11-05", "2025-03-26", "2025-06-18")
DEFAULT_VERSION = "2025-06-18"

#: JSON-RPC 2.0 error codes, only the ones this server can produce.
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INTERNAL_ERROR = -32603


def _receipt_words(receipt: dict[str, Any], handle: str) -> str:
    """The receipt rule: a write is only proven by the id (or key) and
    version the server answered with, so the tool result repeats them.
    A server that answered without one says so, rather than passing as
    recorded."""
    ident = receipt.get(handle)
    version = receipt.get("version")
    if ident is None or version is None:
        return " - but the server sent no receipt, so treat it as unproven."
    return f" - receipt: {handle} {ident}, version {version}."


def _guidance_words(entries: list[dict[str, Any]]) -> str:
    """The customer's own instructions about what to record, spelled out in
    the tool result rather than left in a field nobody reads. Silent when
    they have not set any, so a session that has none is not told about a
    feature it is not using."""
    if not entries:
        return ""
    lines = ["", "What this customer wants recorded here:"]
    for entry in entries:
        topic = _string(entry.get("topic"))
        what = _string(entry.get("guidance"))
        why = _string(entry.get("reason"))
        where = _string(entry.get("scope"))
        mark = " (for this session only)" if where == "session" else ""
        lines.append(f"- {topic}{mark}: {what}")
        if why:
            lines.append(f"  Why: {why}")
    lines.append(
        "Watch for these as you work and record them as they come up."
    )
    return "\n".join(lines)


def _string(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _strings(value: Any) -> list[str]:
    """A list of non-empty strings from a tool argument, or nothing."""
    if not isinstance(value, list):
        return []
    return [v.strip() for v in value if isinstance(v, str) and v.strip()]


TOOLS: tuple[dict[str, Any], ...] = (
    {
        "name": "brief",
        "description": (
            "What this project already knows: its facts, the decisions "
            "already settled, what is open, and what happened recently. The "
            "session start hook fetches this automatically; call it again "
            "only if you need it refreshed mid-session."
        ),
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "open_session",
        "description": (
            "Name the work this session is doing. Optional - the first thing "
            "you record opens a session by itself - but a session you name is "
            "easier for the next one to find than one named for you."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "A short name for this session's work.",
                },
                "summary": {
                    "type": "string",
                    "description": "Optional: what it set out to do.",
                },
            },
            "required": ["title"],
        },
    },
    {
        "name": "note_decision",
        "description": (
            "Record a question that has been settled, and why. Use this "
            "whenever a choice is made that a later session would otherwise "
            "reopen - an approach picked over an alternative, a constraint "
            "accepted, a name agreed. The rationale is the part worth "
            "keeping: without it the next session knows what was chosen but "
            "not what was rejected, and reopens the question the first time "
            "the choice looks odd. If the project already holds a decision "
            "on a question like this one, the write is held back and every "
            "such entry is returned: show them all to the person, ask which "
            "is authoritative, and only then call again with 'supersedes' "
            "or 'distinct_from'. Never pick for them."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The question that was open.",
                },
                "supersedes": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Ids of earlier decisions this one replaces - only "
                        "after the person was shown them and said the new "
                        "answer stands. They are kept, marked as replaced."
                    ),
                },
                "distinct_from": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Ids of earlier decisions the person says are "
                        "different questions, so this one is written "
                        "alongside them."
                    ),
                },
                "decision": {
                    "type": "string",
                    "description": "What was settled.",
                },
                "rationale": {
                    "type": "string",
                    "description": "Why, and what was rejected.",
                },
                "source": {
                    "type": "string",
                    "description": (
                        "Required: where this came from - a document, a "
                        "specific message, an earlier session. An entry "
                        "nobody can trace back is an entry nobody can "
                        "check; the database accepts the write without it "
                        "rather than lose the record, and marks it "
                        "'unset:api-did-not-supply-source' so the gap is "
                        "visible. Write the source instead."
                    ),
                },
                "decided_at": {
                    "type": "string",
                    "description": (
                        "ONLY when writing down a decision that was already "
                        "made - backfilling history. ISO 8601 with a "
                        "timezone, e.g. 2026-08-23T14:05:00-07:00: the "
                        "moment it was actually settled, so the project "
                        "holds it as done and over with, not as an open "
                        "question. Leave it out for live work. Write "
                        "'unknown' when you are recording something that was "
                        "settled but you cannot establish when: the decision "
                        "is stored with no date at all and counted as "
                        "undated, which is the truth. Never put today's date "
                        "on something that did not happen today - that is "
                        "what leaving it out would do."
                    ),
                },
            },
            "required": ["question", "decision", "source"],
        },
    },
    {
        "name": "note_fact",
        "description": (
            "Record something durable about the project: a deadline, a "
            "constraint, a name, an address, a convention. Setting the same "
            "key again replaces it, because a fact that changed is not two "
            "facts. Use this for what stays true, not for what just happened. "
            "If the project already holds a different value - under this key "
            "from another session, or under a key that looks like this one - "
            "the write is held back and every such entry is returned: show "
            "them all to the person, ask which is authoritative, and only "
            "then call again with 'supersedes' or 'distinct_from'."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "description": "What this fact is about.",
                },
                "value": {"type": "string", "description": "What is true."},
                "supersedes": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Keys of earlier facts this one replaces - only after "
                        "the person was shown them and said the new value "
                        "stands. This key means overwrite it; another key "
                        "means retire that fact, its value kept in history."
                    ),
                },
                "distinct_from": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Keys of earlier facts the person says are about "
                        "different things, so this one is written alongside."
                    ),
                },
                "source": {
                    "type": "string",
                    "description": (
                        "Required: where this came from - a document, a "
                        "schedule, a specific message. A fact nobody can "
                        "trace back cannot be re-verified when it goes "
                        "stale; the database accepts the write without it "
                        "rather than lose the record, and marks it "
                        "'unset:api-did-not-supply-source'. Write the "
                        "source instead."
                    ),
                },
            },
            "required": ["key", "value", "source"],
        },
    },
    {
        "name": "note_activity",
        "description": (
            "Record something done, so progress is visible across weeks "
            "rather than only inside one session. Call it the moment a real "
            "change lands - a setting edited, a row written, a fix pushed - "
            "not when the surrounding task is finished. Completion is its "
            "own, separate entry."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "What was done.",
                },
                "kind": {
                    "type": "string",
                    "description": (
                        "What this is: 'finding' for something discovered "
                        "(record it before anything else about it), 'fix', "
                        "'build', 'change' or 'note' for work done, 'deploy' "
                        "for a commit, push or live deploy - deploys are kept "
                        "out of the brief's recent list so findings come "
                        "first."
                    ),
                },
                "source": {
                    "type": "string",
                    "description": (
                        "Required: where this came from - the conversation, "
                        "the file, the command whose output this is. The "
                        "database accepts the write without it rather than "
                        "lose the record, and marks it "
                        "'unset:api-did-not-supply-source'."
                    ),
                },
                "occurred_at": {
                    "type": "string",
                    "description": (
                        "ONLY when writing down something that already "
                        "happened - backfilling history. ISO 8601 with a "
                        "timezone: the moment it actually happened. Leave "
                        "it out for live work. Write 'unknown' when you are "
                        "recording something that did happen but you cannot "
                        "establish when: the entry is stored with no date at "
                        "all and counted as undated, which is the truth. "
                        "Never put today's date on something that did not "
                        "happen today - that is what leaving it out would do."
                    ),
                },
            },
            "required": ["description", "source"],
        },
    },
    {
        "name": "find_fact",
        "description": (
            "Look up one fact by its exact key, without re-fetching the "
            "whole brief. Use this when you need to check a single value "
            "mid-session rather than everything the project knows."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "description": "The fact's key, as it was recorded.",
                },
            },
            "required": ["key"],
        },
    },
    {
        "name": "look_up",
        "description": (
            "Look something up across everything the project holds - facts, "
            "decisions, activity, preferences and recording guidance - in one "
            "call, and be told which store each hit came from. Call this "
            "before ever saying something is not recorded: find_fact reads "
            "one store, and a rule recorded as a preference is invisible to "
            "it. Every word you give must appear in the entry; give a few "
            "words from what you remember of it."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A few words from what you are looking for.",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "note_preference",
        "description": (
            "Record how this person wants to be worked with - how they like "
            "to be addressed, the units they think in, the format they want, "
            "words they never want used. A preference follows the person "
            "into every project (scope 'user') unless they say it is for "
            "this project only (scope 'project'). Use source 'stated' when "
            "they said it and 'observed' when you only noticed it - and say "
            "so to them when you record an observed one."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "description": "A short name, such as 'tone' or 'units'.",
                },
                "value": {
                    "type": "string",
                    "description": "The preference itself, in plain words.",
                },
                "scope": {
                    "type": "string",
                    "enum": ["user", "project"],
                    "description": "'user' (default) or 'project'.",
                },
                "source": {
                    "type": "string",
                    "description": "'stated' (default) or 'observed'.",
                },
            },
            "required": ["key", "value"],
        },
    },
    {
        "name": "note_guidance",
        "description": (
            "Record what this person wants Zendavox to record, and why - "
            "the things that matter in their business that no assistant "
            "could guess. Use it whenever they say what should be captured "
            "('always note which system a customer runs, because that is "
            "what I get asked first'), or what one session is for. Three "
            "widths: 'user' follows them into every project, 'project' (the "
            "default) is this project, 'session' is this one piece of work "
            "and ends with it. The narrowest wins on the same topic. Always "
            "ask for the reason if they did not give one - it is what lets "
            "a later session apply the instruction to a case they never "
            "described."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": (
                        "What this is about, in their words - 'customers', "
                        "'the programs I write', 'board meetings'."
                    ),
                },
                "guidance": {
                    "type": "string",
                    "description": "What they want recorded.",
                },
                "reason": {
                    "type": "string",
                    "description": "Why it matters to them.",
                },
                "scope": {
                    "type": "string",
                    "enum": ["user", "project", "session"],
                    "description": "'project' (default), 'user', or 'session'.",
                },
                "source": {
                    "type": "string",
                    "description": "'stated' (default) or 'observed'.",
                },
            },
            "required": ["topic", "guidance"],
        },
    },
    {
        "name": "close_session",
        "description": (
            "Say what this session amounted to, so the next one can skim "
            "instead of reading every activity. Call this before you finish."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "What this session amounted to.",
                }
            },
            "required": ["summary"],
        },
    },
)

def render_lookup(answer: dict[str, Any]) -> str:
    """The lookup as words an assistant can act on: every hit with its
    store, or - when there is none - which stores were searched, so the
    assistant says "not recorded" and not "missing"."""
    query = _string(answer.get("query"))
    searched = answer.get("searched") or []
    stores = ", ".join(str(s) for s in searched) or "every store"
    hits = answer.get("hits") or []
    if not hits:
        return (
            f"Nothing in any store matches {query!r}. All of these were "
            f"searched: {stores}. That is 'not recorded', not 'missing' - "
            "try other words from it before concluding either."
        )
    lines = [f"{len(hits)} hit{'s' if len(hits) != 1 else ''} for {query!r}:"]
    for h in hits:
        store = _string(h.get("store"))
        scope = _string(h.get("scope"))
        where = f"{store} ({scope})" if scope else store
        when = _string(h.get("when"))[:10]
        source = _string(h.get("source"))
        tail = f" [{when}" + (f", source: {source}" if source else "") + "]"
        lines.append(f"- {where}: {_string(h.get('title'))}")
        lines.append(f"  {_string(h.get('text'))}{tail}")
    return "\n".join(lines)


class ToolFailure(RuntimeError):
    """A tool could not do what was asked, in a way worth telling the model."""


@dataclass
class Server:
    """One MCP conversation, for one coding session."""

    root: pathlib.Path
    config: DevConfig | None = None
    client: Client | None = None
    state_dir: pathlib.Path | None = None
    _settings: DevConfig | None = field(default=None, init=False, repr=False)

    # ------------------------------------------------------------------
    # wiring
    # ------------------------------------------------------------------
    def settings(self) -> DevConfig:
        if self._settings is None:
            self._settings = (
                load(root=self.root) if self.config is None else self.config
            )
        return self._settings

    def line(self) -> Client:
        if self.client is not None:
            return self.client
        settings = self.settings()
        if not settings.configured or settings.key is None:
            raise ToolFailure(
                "No Zendavox key is configured, so nothing can be recorded. "
                "Set ZENDAVOX_DEV_KEY, or put a key in ~/.zendavox/dev.json. "
                "Get one from the Connect page at "
                f"{settings.base_url}/connect."
            )
        self.client = Client(
            base_url=settings.base_url,
            key=settings.key,
            timeout=settings.timeout,
        )
        return self.client

    # ------------------------------------------------------------------
    # the session record, opened only when there is something to put in it
    # ------------------------------------------------------------------
    def _key(self) -> str:
        """Keyed on the checkout, so the end hook can find what we opened."""
        return state.key_for_root(self.root)

    def _current(self) -> state.SessionState:
        return state.read(self._key(), directory=self.state_dir)

    def _remember(self, current: state.SessionState) -> None:
        state.write(self._key(), current, directory=self.state_dir)

    def _default_title(self) -> str:
        return f"Coding session in {self.root.name or 'a project'}"

    def _session_id(self, *, title: str | None = None) -> str:
        """The open session record, opening one if this is the first write.

        A *named* call for different work gets its own record. The state is
        keyed on the checkout, not on the conversation (see
        :mod:`zendavox.dev.state`), so a session left open by an earlier
        conversation is still sitting there when the next one starts - and
        everything the new conversation recorded used to be filed under the
        old one's name, while the tool reported back the name it had been
        given. Recorded, and under a title that described something else.
        Steven, 2026-09-06, shown a day's work filed under a session called
        "Marketing: Zendavox feature list for individuals": a new chat that
        names its work should get a session under that name.

        An unnamed write still joins whatever is open - that is the
        deliberate trade in state.py, and a write with no name has nothing
        better to say about itself.
        """
        current = self._current()
        if current.session_id is not None and not current.closed:
            if title is None or title == current.title:
                return current.session_id
        chosen = title or self._default_title()
        session_id = self.line().open_session(title=chosen)
        self._remember(
            state.SessionState(session_id=session_id, title=chosen)
        )
        return session_id

    # ------------------------------------------------------------------
    # tools
    # ------------------------------------------------------------------
    def tools(self) -> tuple[dict[str, Any], ...]:
        """What this server offers. Every tool here is about the project's
        own record, which is what a coding session needs and all it needs;
        a server whose customers work in a chat rather than a checkout adds
        its own - see :class:`zendavox.mcp_http.HostedServer`. Handing a
        developer a board-minutes tool, or a nonprofit's secretary a tool
        for a module they are not paying for, is how a tool list stops
        being read."""
        return TOOLS

    def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        if name not in {tool["name"] for tool in self.tools()}:
            raise ToolFailure(f"There is no Zendavox tool called {name!r}.")

        if name == "brief":
            return render(self.line().brief())

        if name == "close_session":
            summary = _string(arguments.get("summary"))
            if not summary:
                raise ToolFailure("'summary' is required to close a session.")
            current = self._current()
            if current.session_id is None:
                # Nothing was recorded, so there is no record to close. Saying
                # so is more useful than inventing an empty session to close.
                return (
                    "Nothing was recorded this session, so there is no "
                    "session record to close."
                )
            if current.closed:
                return "This session's record is already closed."
            swept = any(
                _string(a.get("kind")) == "sweep"
                for a in self.line().session_activity(current.session_id)
            )
            if not swept:
                # A session used to be able to close with real conversation
                # left unrecorded and nothing anywhere to show it - the gap
                # Steven found on 2026-09-05. Checked before closing, not
                # after, because after is too late to matter: this session
                # is about to end either way.
                raise ToolFailure(
                    "Before closing, check this conversation against the "
                    "brief for anything not yet recorded, then call "
                    "note_activity once with kind \"sweep\" describing what "
                    "you found - or that there was nothing new - and call "
                    "close_session again."
                )
            self.line().close_session(
                session_id=current.session_id, summary=summary
            )
            self._remember(
                state.SessionState(
                    session_id=current.session_id,
                    title=current.title,
                    closed=True,
                )
            )
            return "Session closed. The next session will start from this."

        if name == "open_session":
            title = _string(arguments.get("title"))
            if not title:
                raise ToolFailure("'title' is required to open a session.")
            continuing = self._current()
            session_id = self._session_id(title=title)
            summary = _string(arguments.get("summary"))
            if summary:
                self.line().note_activity(
                    description=summary,
                    kind="note",
                    session_id=session_id,
                )
            # Say which record this actually is. "Recording under X" while
            # writing into Y is the untruth this reply used to tell.
            opening = (
                f"Continuing the open session {title!r}."
                if continuing.session_id == session_id
                else f"Recording under {title!r}."
            )
            # What this customer wants captured, including anything set for
            # this one session - which the brief could not have carried,
            # since it was read before the session existed.
            return opening + _guidance_words(
                self.line().recording_guidance(session_id=session_id)
            )

        # Everything below records against the session. Arguments are checked
        # first, deliberately: opening a session record and only then finding
        # the call was malformed would leave an empty session behind for every
        # mistyped tool call.
        if name == "note_decision":
            question = _string(arguments.get("question"))
            decision = _string(arguments.get("decision"))
            if not question or not decision:
                raise ToolFailure(
                    "'question' and 'decision' are both required."
                )
            receipt = self.line().note_decision(
                question=question,
                decision=decision,
                rationale=_string(arguments.get("rationale")) or None,
                source=_string(arguments.get("source")) or None,
                decided_at=_string(arguments.get("decided_at")) or None,
                session_id=self._session_id(),
                supersedes=_strings(arguments.get("supersedes")),
                distinct_from=_strings(arguments.get("distinct_from")),
            )
            replaced = receipt.get("supersedes") or []
            noted = (
                f", replacing {len(replaced)} earlier decision"
                + ("" if len(replaced) == 1 else "s")
                if replaced
                else ""
            )
            return "Decision recorded" + noted + _receipt_words(receipt, "id")

        if name == "note_fact":
            key = _string(arguments.get("key"))
            value = _string(arguments.get("value"))
            if not key or not value:
                raise ToolFailure("'key' and 'value' are both required.")
            # A fact is durably true about the project, so setting the same
            # key again replaces it rather than making a second one - but it
            # is still tagged with the session that (last) touched it, and a
            # session doing only this should show up as having happened.
            receipt = self.line().note_fact(
                key=key,
                value=value,
                source=_string(arguments.get("source")) or None,
                session_id=self._session_id(),
                supersedes=_strings(arguments.get("supersedes")),
                distinct_from=_strings(arguments.get("distinct_from")),
            )
            retired = [k for k in (receipt.get("supersedes") or []) if k != key]
            noted = (
                f", retiring {len(retired)} earlier fact"
                + ("" if len(retired) == 1 else "s")
                if retired
                else ""
            )
            return f"Fact recorded under {key!r}" + noted + _receipt_words(
                receipt, "key"
            )

        if name == "note_preference":
            key = _string(arguments.get("key"))
            value = _string(arguments.get("value"))
            if not key or not value:
                raise ToolFailure("'key' and 'value' are both required.")
            receipt = self.line().note_preference(
                key=key,
                value=value,
                scope=_string(arguments.get("scope")) or "user",
                source=_string(arguments.get("source")) or "stated",
            )
            return f"Preference recorded under {key!r}" + _receipt_words(
                receipt, "key"
            )

        if name == "note_guidance":
            topic = _string(arguments.get("topic"))
            what = _string(arguments.get("guidance"))
            if not topic or not what:
                raise ToolFailure("'topic' and 'guidance' are both required.")
            scope = _string(arguments.get("scope")) or "project"
            # Guidance for one session needs the session to exist first;
            # for the other two widths it must not open one, since saying
            # what you want recorded is not itself work being done.
            receipt = self.line().note_guidance(
                topic=topic,
                guidance=what,
                reason=_string(arguments.get("reason")) or None,
                scope=scope,
                session_id=self._session_id() if scope == "session" else None,
                source=_string(arguments.get("source")) or "stated",
            )
            return f"Guidance recorded under {topic!r}" + _receipt_words(
                receipt, "topic"
            )

        if name == "look_up":
            query = _string(arguments.get("query"))
            if not query:
                raise ToolFailure("'query' is required.")
            return render_lookup(self.line().look_up(query))

        if name == "find_fact":
            key = _string(arguments.get("key"))
            if not key:
                raise ToolFailure("'key' is required.")
            found = self.line().fact(key)
            fact_value = found.get("value")
            fact_source = found.get("source")
            return f"{key}: {fact_value}" + (
                f" (source: {fact_source})" if fact_source else ""
            )

        description = _string(arguments.get("description"))
        if not description:
            raise ToolFailure("'description' is required.")
        receipt = self.line().note_activity(
            description=description,
            kind=_string(arguments.get("kind")) or "note",
            occurred_at=_string(arguments.get("occurred_at")) or None,
            session_id=self._session_id(),
            source=_string(arguments.get("source")) or None,
        )
        return "Activity recorded" + _receipt_words(receipt, "id")

    # ------------------------------------------------------------------
    # JSON-RPC
    # ------------------------------------------------------------------
    def handle(self, message: dict[str, Any]) -> dict[str, Any] | None:
        """Answer one message. ``None`` means "this was a notification"."""
        method = message.get("method")
        has_id = "id" in message and message["id"] is not None
        message_id = message.get("id")

        if not isinstance(method, str):
            if not has_id:
                return None
            return _error(message_id, INVALID_REQUEST, "No method was named.")

        # Notifications get no reply at all. Replying to one is a protocol
        # violation that some clients treat as a fatal stream error.
        if not has_id:
            return None

        if method == "initialize":
            return self._initialize(message_id, message.get("params"))
        if method == "tools/list":
            return _result(message_id, {"tools": list(self.tools())})
        if method == "tools/call":
            return self._tools_call(message_id, message.get("params"))
        if method == "ping":
            return _result(message_id, {})

        return _error(
            message_id, METHOD_NOT_FOUND, f"Unknown method {method!r}."
        )

    def _initialize(
        self, message_id: Any, params: Any
    ) -> dict[str, Any]:
        asked = ""
        if isinstance(params, dict):
            asked = _string(params.get("protocolVersion"))
        version = asked if asked in KNOWN_VERSIONS else DEFAULT_VERSION
        from zendavox.dev import __version__

        return _result(
            message_id,
            {
                "protocolVersion": version,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": SERVER_NAME, "version": __version__},
                "instructions": (
                    "Zendavox keeps this project's memory between sessions. "
                    "Call `brief` first, before doing anything else - it "
                    "holds what earlier sessions established and your "
                    "complete, current operating rules. Read its rules in "
                    "full and follow them exactly; do not rely on this "
                    "message for anything past this one instruction, since "
                    "the rules change and this text does not."
                ),
            },
        )

    def _tools_call(self, message_id: Any, params: Any) -> dict[str, Any]:
        if not isinstance(params, dict):
            return _tool_error(message_id, "No tool was named.")
        name = _string(params.get("name"))
        raw_arguments = params.get("arguments")
        arguments = raw_arguments if isinstance(raw_arguments, dict) else {}
        if not name:
            return _tool_error(message_id, "No tool was named.")

        try:
            text = self.call_tool(name, arguments)
        except (ToolFailure, DevError) as exc:
            # A tool that could not run is a result the model can act on, not
            # a broken connection. Reporting it as a protocol error would end
            # the conversation over a missing argument.
            return _tool_error(message_id, str(exc))
        except Exception as exc:  # noqa: BLE001 - the stream must survive
            print(
                f"[zendavox-dev] {name} failed: {type(exc).__name__}: {exc}",
                file=sys.stderr,
                flush=True,
            )
            return _tool_error(
                message_id, "Something went wrong recording that."
            )
        return _result(
            message_id,
            {"content": [{"type": "text", "text": text}], "isError": False},
        )


def _result(message_id: Any, payload: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": message_id, "result": payload}


def _error(message_id: Any, code: int, message: str) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": message_id,
        "error": {"code": code, "message": message},
    }


def _tool_error(message_id: Any, message: str) -> dict[str, Any]:
    return _result(
        message_id,
        {"content": [{"type": "text", "text": message}], "isError": True},
    )


def serve(
    server: Server, *, stdin: TextIO | None = None, stdout: TextIO | None = None
) -> int:
    """Run the stdio loop until the far end closes it."""
    source = sys.stdin if stdin is None else stdin
    sink = sys.stdout if stdout is None else stdout

    for raw in source:
        if not raw.strip():
            continue
        try:
            message = json.loads(raw)
        except json.JSONDecodeError:
            _emit(sink, _error(None, PARSE_ERROR, "That is not valid JSON."))
            continue
        if not isinstance(message, dict):
            _emit(
                sink,
                _error(None, INVALID_REQUEST, "A message must be an object."),
            )
            continue
        try:
            reply = server.handle(message)
        except Exception as exc:  # noqa: BLE001 - the stream must survive
            print(
                f"[zendavox-dev] handler failed: {exc!r}",
                file=sys.stderr,
                flush=True,
            )
            reply = _error(
                message.get("id"), INTERNAL_ERROR, "Something went wrong."
            )
        if reply is not None:
            _emit(sink, reply)
    return 0


def _emit(sink: TextIO, message: dict[str, Any]) -> None:
    sink.write(json.dumps(message) + "\n")
    sink.flush()
