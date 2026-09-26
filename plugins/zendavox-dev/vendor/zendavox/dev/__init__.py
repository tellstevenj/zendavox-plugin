"""zendavox-dev: the same continuity, for a coding session.

Zendavox exists because work that spans sessions loses its thread. The product
solves that for a customer's project: a session opens by reading what the
project already knows, records decisions and facts as it goes, and closes with
a summary the next session can skim.

A coding session has exactly the same problem and, until this package, none of
the cure. An assistant opening this repository started from nothing: it read
the files, guessed at the history, and re-derived conclusions that had already
been reached and paid for. That is the failure the product was built to end,
happening inside the tools used to build the product.

This package closes that gap. It is a client for the connector
(:mod:`zendavox.connector_http`), shaped for the two ways a coding assistant
can be reached:

``hooks``
    Run by the harness, not by the assistant. The session-start hook fetches
    the project's brief and hands it to the session before the first prompt,
    so continuity does not depend on anyone remembering to ask for it.

``mcp``
    Tools the assistant calls while it works - record a decision, a fact, an
    activity, close the session. Recognising "that was a decision" needs
    judgement, so this half is deliberately model-driven.

The split is the point, and it is the same one :mod:`zendavox.connector` makes:
reading is automatic because a forgotten read is the whole problem, and writing
is judged because only the session knows what just happened.

**Nothing here is allowed to break a session.** Every entry point run by the
harness fails soft: no key, no network, a bad reply from the far end - all of
them end in a short note and a zero exit code. An assistant that cannot reach
Zendavox should work exactly as it did before, which is to say with no memory,
rather than not start at all.
"""

from __future__ import annotations

#: Bumped when the shape of what the hooks emit changes.
__version__ = "1.0.0"
