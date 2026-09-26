"""Talking to the connector from outside the product.

:mod:`zendavox.connector_http` is the far side of this: JSON in, JSON out, a
key in an ``Authorization: Bearer`` header. This is the near side, in the
standard library only, so a hook or an MCP server can run wherever a Python
interpreter does without an install step first.

Two rules shape the whole module.

**Every failure is one exception.** A missing key, an unreachable host, a 500
from the far end - the caller gets a :class:`DevError` with a sentence it can
show a person. Callers here are a hook that must not break a session and a
tool that must not break a turn; neither is in a position to handle six kinds
of failure differently, and both need a sentence to print.

**The transport is injectable.** Every test in this repository that needs the
network is skipped without a live database, which means an untested client is
an untestable client. Passing the transport in makes the request-building and
reply-handling - where the bugs actually are - testable with no network at all.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import quote, urlencode, urlparse

from zendavox.dev.config import DEFAULT_BASE_URL

#: Deliberately repeated from :data:`zendavox.connector_http.API_ROOT` rather
#: than imported from it. That module reaches the database and so imports
#: SQLAlchemy, and importing it here would put a compiled dependency between a
#: session-start hook and the interpreter it runs under - on every machine, for
#: the sake of five characters. The copy is held to the real value by a test
#: that imports both and compares them.
API_ROOT = "/api/v1"

#: ``(method, url, body, headers, timeout) -> (status, text)``.
Transport = Callable[
    [str, str, str | None, dict[str, str], float], tuple[int, str]
]

USER_AGENT = "zendavox-dev"


class DevError(RuntimeError):
    """Something went wrong that the caller should say out loud."""


class Unreachable(DevError):
    """Nothing answered at that address at all.

    A subclass rather than a flag, so every existing caller that catches
    :class:`DevError` is unaffected. The distinction matters in exactly one
    place: a sleeping web service holds the connection open while it boots,
    whereas a closed port refuses immediately. Waiting out a boot is right;
    waiting ninety seconds on a refusal is a hook that hangs a session.
    """


def _http(
    method: str,
    url: str,
    body: str | None,
    headers: dict[str, str],
    timeout: float,
) -> tuple[int, str]:
    """The real transport. Any HTTP status is a result, not an exception -
    the connector says what is wrong in the body of a 400, and a transport
    that raised on it would throw that sentence away."""
    data = None if body is None else body.encode("utf-8")
    request = urllib.request.Request(
        url, data=data, headers=headers, method=method
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return int(response.status), response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        return int(exc.code), exc.read().decode("utf-8", errors="replace")
    except urllib.error.URLError as exc:
        # A connect timeout arrives wrapped in URLError rather than as a bare
        # TimeoutError, so unwrap it before deciding which of the two this is.
        if isinstance(exc.reason, TimeoutError):
            raise DevError("Zendavox did not answer in time.") from exc
        raise Unreachable(f"Could not reach Zendavox: {exc.reason}") from exc
    except TimeoutError as exc:
        raise DevError("Zendavox did not answer in time.") from exc
    except OSError as exc:
        raise Unreachable(f"Could not reach Zendavox: {exc}") from exc


@dataclass(frozen=True)
class Client:
    """A session's line to its project's record."""

    base_url: str
    key: str
    timeout: float = 12.0
    transport: Transport = field(default=_http, repr=False)

    def _request(
        self,
        method: str,
        route: str,
        body: dict[str, Any] | None,
        *,
        query: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        url = f"{self.base_url}{API_ROOT}{route}"
        if query:
            url = f"{url}?{urlencode(query)}"
        scheme = urlparse(url).scheme
        if scheme not in ("http", "https"):
            raise DevError(f"{self.base_url!r} is not a usable address.")

        headers = {
            "Authorization": f"Bearer {self.key}",
            "Accept": "application/json",
            "User-Agent": USER_AGENT,
        }
        payload = None
        if body is not None:
            payload = json.dumps(body)
            headers["Content-Type"] = "application/json"

        status, text = self.transport(
            method, url, payload, headers, self.timeout
        )

        try:
            parsed = json.loads(text) if text.strip() else {}
        except json.JSONDecodeError:
            # The application always answers JSON, so a non-JSON body means
            # something in front of it answered instead - a proxy or a web
            # application firewall. Saying "the address may be wrong" sent a
            # session hunting a configuration problem for twenty minutes on
            # 2026-09-11 when the address was right and a firewall rule had
            # rejected one sentence of a fact's text.
            hint = (
                "something in front of Zendavox refused the request rather "
                "than the application answering. The address is probably "
                "fine. If this is a write, the text may have tripped a "
                "firewall rule - try rewording it."
                if 400 <= status < 500
                else "the address may be wrong, or the host is not serving "
                "the application."
            )
            raise DevError(
                f"Zendavox answered {status} with something that is not "
                f"JSON: {hint}"
            ) from None
        if not isinstance(parsed, dict):
            raise DevError(f"Zendavox answered {status} with unexpected JSON.")

        if status == 200:
            return parsed

        # A redirect is not an answer. urllib follows one for GET, so reads
        # look fine, but it will not re-send a POST after a 307, so every
        # write dies here with nothing in the body to explain it. Say what
        # happened and what fixes it - the live site is on www, and the bare
        # domain only redirects to it.
        if status in (301, 302, 303, 307, 308) and not parsed.get("error"):
            raise DevError(
                f"Zendavox redirected the request ({status}) instead of "
                f"answering it, so {self.base_url!r} is not where the API "
                "lives. Set ZENDAVOX_DEV_URL to the address it redirects to "
                f"(the live site is {DEFAULT_BASE_URL})."
            )

        # The connector puts a caller-fixable sentence in 'error'. Prefer it
        # over anything invented here: it knows what went wrong, we do not.
        message = str(parsed.get("error") or f"Zendavox answered {status}.")
        reference = parsed.get("reference")
        if reference:
            message = f"{message} (reference {reference})"
        raise DevError(message)

    def _call(
        self,
        route: str,
        body: dict[str, Any] | None,
        *,
        query: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Reads over GET, writes over POST - the split the connector's own
        do_GET comment describes. A GET carries no body at all."""
        return self._request(
            "GET" if body is None else "POST", route, body, query=query
        )

    def _delete(self, route: str) -> dict[str, Any]:
        return self._request("DELETE", route, None)

    # ------------------------------------------------------------------
    # what a session records as it works
    # ------------------------------------------------------------------
    def brief(self) -> dict[str, Any]:
        """What this project already knows. The first call a session makes."""
        return self._call("/brief", None)

    def status(self) -> dict[str, Any]:
        """The project's version, generated_at, and record counts - cheap
        enough to poll before deciding whether ``brief()`` or ``changes()``
        is worth calling at all."""
        return self._call("/status", None)

    def changes(
        self, *, since: int, limit: int | None = None
    ) -> dict[str, Any]:
        """Everything recorded after ``since`` - the delta a session applies
        to what it already has instead of asking for the whole brief again."""
        query = {"since": str(since)}
        if limit is not None:
            query["limit"] = str(limit)
        return self._call("/changes", None, query=query)

    def open_session(self, *, title: str, summary: str | None = None) -> str:
        body: dict[str, Any] = {"title": title}
        if summary:
            body["summary"] = summary
        result = self._call("/session/open", body)
        session_id = result.get("session_id")
        if not isinstance(session_id, str) or not session_id:
            raise DevError("Zendavox opened a session but did not name it.")
        return session_id

    def close_session(self, *, session_id: str, summary: str) -> None:
        self._call(
            "/session/close", {"session_id": session_id, "summary": summary}
        )

    def note_decision(
        self,
        *,
        question: str,
        decision: str,
        rationale: str | None = None,
        session_id: str | None = None,
        source: str | None = None,
        decided_at: str | None = None,
        supersedes: list[str] | None = None,
        distinct_from: list[str] | None = None,
    ) -> dict[str, Any]:
        """``decided_at`` (ISO 8601, with a timezone) is only for writing
        down a decision that was already made - a backfill. Leave it out
        for live work and the server stamps now. Returns the server's
        receipt: the decision's ``id`` and the project ``version``.

        A live decision that collides with one the project already holds
        is refused (HTTP 409, surfacing here as a :class:`DevError` whose
        message lists every colliding entry). ``supersedes`` and
        ``distinct_from`` carry the person's answer back."""
        body: dict[str, Any] = {"question": question, "decision": decision}
        if rationale:
            body["rationale"] = rationale
        if session_id:
            body["session_id"] = session_id
        if source:
            body["source"] = source
        if decided_at:
            body["decided_at"] = decided_at
        if supersedes:
            body["supersedes"] = list(supersedes)
        if distinct_from:
            body["distinct_from"] = list(distinct_from)
        return self._call("/decision", body)

    def note_fact(
        self,
        *,
        key: str,
        value: str,
        source: str | None = None,
        session_id: str | None = None,
        supersedes: list[str] | None = None,
        distinct_from: list[str] | None = None,
    ) -> dict[str, Any]:
        """Returns the receipt: the fact's ``key`` and the project
        ``version`` it became part of. A value that collides with what the
        project holds is refused (HTTP 409, a :class:`DevError` listing
        every colliding fact); ``supersedes`` and ``distinct_from`` carry
        the person's answer back."""
        body: dict[str, Any] = {"key": key, "value": value}
        if source:
            body["source"] = source
        if session_id:
            body["session_id"] = session_id
        if supersedes:
            body["supersedes"] = list(supersedes)
        if distinct_from:
            body["distinct_from"] = list(distinct_from)
        return self._call("/fact", body)

    def note_activity(
        self,
        *,
        description: str,
        kind: str = "note",
        session_id: str | None = None,
        source: str | None = None,
        occurred_at: str | None = None,
    ) -> dict[str, Any]:
        """``occurred_at`` (ISO 8601, with a timezone) is only for writing
        down something that already happened - a backfill. Returns the
        receipt: the activity's ``id`` and the project ``version``."""
        body: dict[str, Any] = {"description": description, "kind": kind}
        if occurred_at:
            body["occurred_at"] = occurred_at
        if session_id:
            body["session_id"] = session_id
        if source:
            body["source"] = source
        return self._call("/activity", body)

    # ------------------------------------------------------------------
    # reading a single record back, and removing one
    # ------------------------------------------------------------------
    def facts(self) -> list[dict[str, Any]]:
        """Every fact recorded for the project."""
        result = self._call("/facts", None)
        facts = result.get("facts")
        return facts if isinstance(facts, list) else []

    def note_preference(
        self, *, key: str, value: str, scope: str = "user", source: str = "stated"
    ) -> dict[str, Any]:
        """How the person wants to be worked with. Returns the receipt: the
        preference's ``key``, its ``scope``, and the project ``version``."""
        return self._call(
            "/preference",
            {"key": key, "value": value, "scope": scope, "source": source},
        )

    def preferences(self) -> list[dict[str, Any]]:
        """How the key holder wants to be worked with, resolved for this
        project (a project override beats their global setting)."""
        found = self._call("/preferences", None).get("preferences", [])
        return list(found) if isinstance(found, list) else []

    def note_guidance(
        self,
        *,
        topic: str,
        guidance: str,
        reason: str | None = None,
        scope: str = "project",
        session_id: str | None = None,
        source: str = "stated",
    ) -> dict[str, Any]:
        """What this customer wants recorded, and why. Returns the receipt:
        the ``topic``, its ``scope``, and the project ``version``."""
        body: dict[str, Any] = {
            "topic": topic,
            "guidance": guidance,
            "scope": scope,
            "source": source,
        }
        if reason:
            body["reason"] = reason
        if session_id:
            body["session_id"] = session_id
        return self._call("/guidance", body)

    def recording_guidance(
        self, *, session_id: str | None = None
    ) -> list[dict[str, Any]]:
        """What to watch for here, already resolved - the person's standing
        guidance, then the project's, then (given a session) that session's
        own, each replacing the wider one on the same topic."""
        query = {"session_id": session_id} if session_id else None
        found = self._call("/guidance", None, query=query).get(
            "recording_guidance", []
        )
        return list(found) if isinstance(found, list) else []

    def forget_guidance(
        self, *, topic: str, scope: str = "project", session_id: str | None = None
    ) -> dict[str, Any]:
        if scope == "session":
            if not session_id:
                raise DevError(
                    "Forgetting guidance for one session needs its id."
                )
            return self._delete(
                f"/guidance/session/{quote(session_id, safe='')}"
                f"/{quote(topic, safe='')}"
            )
        return self._delete(
            f"/guidance/{quote(scope, safe='')}/{quote(topic, safe='')}"
        )

    def look_up(self, query: str) -> dict[str, Any]:
        """Every store at once - facts, decisions, activity, preferences,
        recording guidance - each hit saying which store it came from. The
        call to make before saying anything is not recorded."""
        return self._call("/lookup", None, query={"q": query})

    def fact(self, key: str) -> dict[str, Any]:
        """One fact by its key. Raises :class:`DevError` if nothing is
        recorded under it - the same "one failure, one exception" rule as
        everywhere else in this client."""
        return self._call(f"/fact/{quote(key, safe='')}", None)

    def fact_history(self, key: str) -> list[dict[str, Any]]:
        """Every value this fact used to hold, most recently replaced first."""
        result = self._call(f"/fact/{quote(key, safe='')}/history", None)
        history = result.get("history")
        return history if isinstance(history, list) else []

    def forget_fact(self, key: str) -> bool:
        """Remove a fact. Its last value stays in the fact's history."""
        result = self._delete(f"/fact/{quote(key, safe='')}")
        return bool(result.get("deleted"))

    def forget_decision(self, decision_id: str) -> bool:
        result = self._delete(f"/decision/{quote(decision_id, safe='')}")
        return bool(result.get("deleted"))

    def forget_activity(self, activity_id: str) -> bool:
        result = self._delete(f"/activity/{quote(activity_id, safe='')}")
        return bool(result.get("deleted"))

    def sessions(self) -> list[dict[str, Any]]:
        """Every session in the project, most recently active first."""
        result = self._call("/sessions", None)
        found = result.get("sessions")
        return found if isinstance(found, list) else []

    def session_facts(self, session_id: str) -> list[dict[str, Any]]:
        result = self._call(f"/session/{quote(session_id, safe='')}/facts", None)
        facts = result.get("facts")
        return facts if isinstance(facts, list) else []

    def session_activity(self, session_id: str) -> list[dict[str, Any]]:
        result = self._call(f"/session/{quote(session_id, safe='')}/activity", None)
        found = result.get("activity")
        return found if isinstance(found, list) else []

    def session_decisions(self, session_id: str) -> list[dict[str, Any]]:
        result = self._call(f"/session/{quote(session_id, safe='')}/decisions", None)
        found = result.get("decisions")
        return found if isinstance(found, list) else []
