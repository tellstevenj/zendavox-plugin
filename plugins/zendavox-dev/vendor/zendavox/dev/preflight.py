"""Wake the service, and say which half is asleep when one is.

Free hosting sleeps in two places and they fail differently, which is the
whole reason this module exists rather than a longer timeout:

* **The web service** spins down after minutes of quiet and takes tens of
  seconds to boot. Traffic wakes it. Waiting is the correct response.
* **The database** pauses after days of inactivity on Supabase's free plan and
  does **not** come back on its own. Waiting is useless; somebody has to
  restore it from the dashboard. Only Steven can do that.

Telling those apart is not guesswork. ``/health`` on the site already answers
both halves in one unauthenticated request - 200 ``ok`` when the database
answers, 503 and the reason when it does not - precisely so a caller can tell
"asleep" from "broken". This module is the client side of that.

Steven, 2026-09-11, after a session opened with "Zendavox is not connected"
while the record was in fact healthy: figure out what caused the timeout and
have a routine that checks the service is up and awake before reading or
writing anything. The cause was the 12-second default giving up during a cold
start, and this is that routine.

**It never raises.** A caller in a hook cannot afford an exception, so every
path returns a :class:`Preflight` describing what it found.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from zendavox.dev.client import Client, DevError, Transport, Unreachable, _http
from zendavox.dev.config import DevConfig

#: Long enough for a cold Render container, short enough that a truly dead
#: host does not hold a session open indefinitely.
DEFAULT_DEADLINE = 90.0

#: What a hook may spend. A harness kills a hook that overruns - Claude Code's
#: own limit is sixty seconds - and a killed hook tells the session nothing at
#: all, which is worse than a short wait that gives up politely.
HOOK_DEADLINE = 45.0

#: One probe's own timeout. Deliberately short: a sleeping host does not
#: answer at all, so a long per-request wait just delays the next attempt.
PROBE_TIMEOUT = 10.0

#: A paused database will not wake from traffic, so probe it only a couple of
#: times in case the failure was transient, then report rather than wait.
DATABASE_ATTEMPTS = 3

#: A refused connection means nothing is listening, which waiting cannot
#: change. Two attempts in case a name lookup was briefly unlucky, then stop.
REFUSALS_ALLOWED = 2

READY = "ready"
HOST_ASLEEP = "host_asleep"
DATABASE_ASLEEP = "database_asleep"
MISCONFIGURED = "misconfigured"
UNAUTHORISED = "unauthorised"
NO_KEY = "no_key"
#: The probe itself broke, rather than finding anything out. Its own state
#: because "never raises" has to hold even when the transport misbehaves, and
#: reporting a bug here as a sleeping host would send somebody to wait for a
#: service that is fine.
PROBE_FAILED = "probe_failed"


@dataclass(frozen=True)
class Preflight:
    """What a probe of the service found."""

    state: str
    detail: str
    waited: float = 0.0
    attempts: int = 0
    version: str = ""

    @property
    def ok(self) -> bool:
        return self.state == READY

    @property
    def needs_a_person(self) -> bool:
        """Whether waiting or retrying cannot fix this."""
        return self.state in (
            DATABASE_ASLEEP,
            MISCONFIGURED,
            UNAUTHORISED,
            NO_KEY,
            PROBE_FAILED,
        )


def _probe(
    base_url: str, route: str, timeout: float, transport: Transport
) -> tuple[int, str]:
    return transport(
        "GET",
        f"{base_url}{route}",
        None,
        {"User-Agent": "zendavox-dev/preflight"},
        timeout,
    )


def warm(
    config: DevConfig,
    *,
    deadline: float = DEFAULT_DEADLINE,
    transport: Transport | None = None,
    client: Client | None = None,
    sleep: object = time.sleep,
    now: object = time.monotonic,
) -> Preflight:
    """Get the service ready, or explain precisely why it is not.

    Polls ``/health`` with backoff until it answers 200, the database is
    reported unreachable, or ``deadline`` passes. On success it confirms the
    key with ``/status``, which is the cheapest authenticated call there is -
    so a caller that gets ``ok`` back can read or write without a second
    thought.

    A caller that already holds a :class:`Client` should pass it: the probes
    then go down that client's own transport, which is one line to the service
    rather than two and means a caller testing with a fake client is really
    testing with a fake client.
    """
    if not config.configured or config.key is None:
        return Preflight(
            NO_KEY,
            "No key is configured for this folder, so there is nothing to "
            "wake. Put one in the folder's .env as ZENDAVOX_DEV_KEY.",
        )

    if transport is None:
        transport = client.transport if client is not None else _http

    assert callable(sleep) and callable(now)
    started = now()
    attempts = 0
    database_failures = 0
    refusals = 0
    last = "the host did not answer"
    pause = 1.0

    while now() - started < deadline:
        attempts += 1
        try:
            status, body = _probe(
                config.base_url, "/health", PROBE_TIMEOUT, transport
            )
        except Unreachable as exc:
            refusals += 1
            last = str(exc)
            if refusals >= REFUSALS_ALLOWED:
                return Preflight(
                    HOST_ASLEEP,
                    "Nothing is listening at that address, which is a refusal "
                    "rather than a slow start, so waiting will not help. "
                    f"Check ZENDAVOX_DEV_URL. The machine said: {last}",
                    now() - started,
                    attempts,
                )
            status, body = 0, ""
        except DevError as exc:
            last = str(exc)
            status, body = 0, ""
        except Exception as exc:  # noqa: BLE001 - this may not raise, ever
            return Preflight(
                PROBE_FAILED,
                "The wake-up check itself failed, which says nothing about "
                f"the service: {exc!r}",
                now() - started,
                attempts,
            )

        text = body.strip().lower()
        # Any 200 means both halves answered: the endpoint returns 200 only
        # after the database has answered a query. Matching on the body would
        # make this brittle for nothing, and the key check below is what
        # actually proves the record is readable.
        if status == 200:
            break
        if status == 503 and "no database configured" in text:
            return Preflight(
                MISCONFIGURED,
                "The site is awake but has no database configured, which is a "
                "server setting rather than anything on this machine.",
                now() - started,
                attempts,
            )
        if status == 503 and "database unreachable" in text:
            database_failures += 1
            last = body.strip()
            if database_failures >= DATABASE_ATTEMPTS:
                return Preflight(
                    DATABASE_ASLEEP,
                    "The site is awake but the database is not answering, and "
                    "a paused database does not come back on its own. On "
                    "Supabase's free plan a project pauses after days of "
                    "inactivity and has to be restored from the dashboard. "
                    f"The site said: {last}",
                    now() - started,
                    attempts,
                )
        elif status:
            last = f"the site answered {status}: {body.strip()[:120]}"

        sleep(min(pause, 8.0))
        pause *= 2
    else:
        return Preflight(
            HOST_ASLEEP,
            "The site never answered, so it is either still booting or down. "
            f"Last thing it said: {last}",
            now() - started,
            attempts,
        )

    version = ""
    try:
        code, text = _probe(
            config.base_url, "/version", PROBE_TIMEOUT, transport
        )
        if code == 200:
            version = text.strip()
    except Exception:  # noqa: BLE001 - a build number is never worth failing
        pass  # nice to have, and the record is readable either way

    line = client if client is not None else Client(
        base_url=config.base_url, key=config.key, timeout=config.timeout
    )
    try:
        line.status()
    except Exception as exc:  # noqa: BLE001 - this may not raise, ever
        if not isinstance(exc, DevError):
            return Preflight(
                PROBE_FAILED,
                "The site and database are both up, but the key check itself "
                f"failed rather than being refused: {exc!r}",
                now() - started,
                attempts,
                version,
            )
        message = str(exc)
        if "401" in message or "unauthor" in message.lower():
            return Preflight(
                UNAUTHORISED,
                "The site and database are both up, but this key was refused. "
                "It may have been revoked or replaced. "
                f"The site said: {message}",
                now() - started,
                attempts,
                version,
            )
        return Preflight(
            HOST_ASLEEP,
            "The site's health check passed but the record would not answer: "
            f"{message}",
            now() - started,
            attempts,
            version,
        )

    return Preflight(
        READY,
        "Site and database both answered; the key works.",
        now() - started,
        attempts,
        version,
    )


def summarise(result: Preflight) -> str:
    """One line for the operator, in plain words."""
    where = f"after {result.waited:.0f}s and {result.attempts} probe(s)"
    if result.ok:
        build = f", running {result.version}" if result.version else ""
        return f"service ready {where}{build}."
    return f"{result.state.replace('_', ' ')} {where}: {result.detail}"
