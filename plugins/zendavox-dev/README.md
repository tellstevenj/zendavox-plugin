# zendavox-dev

Zendavox for Claude Code. It gives each coding session the memory of the ones
before it, and keeps a chat on the subject it started on.

- **At the start of every session** it reads your Zendavox project's brief -
  what earlier sessions decided, found and left open - and hands it to Claude
  before your first message.
- **While you work** Claude records decisions, facts and findings into the
  project through the `zendavox` tools.
- **At the end** it makes sure the session is closed with a summary, so the
  next one can pick up where this one stopped.
- **Before each message** it checks whether the chat has drifted away from
  what it started on. After three messages in a row about something else, a
  short "Topic check" line appears; after four, an "Off topic" warning with a
  countdown; at seven, if a small Claude model agrees the subject has really
  changed, the message is held back and you are told why. Start your message
  with `continue:` to carry on anyway.

## Install

You need a Zendavox account and a key for your project: get one at
<https://www.zendavox.com/connect>.

**1. Put the key where the plugin can find it.** Any one of these, first
match wins:

- `ZENDAVOX_DEV_KEY` in your environment,
- a line `ZENDAVOX_DEV_KEY=...` in a `.env` file in your project folder (keep
  `.env` out of git),
- `~/.zendavox/dev.json` containing `{"key": "..."}`, which covers every
  project on the machine.

**2. Add the plugin.** In Claude Code:

```
/plugin marketplace add tellstevenj/zendavox-plugin
/plugin install zendavox-dev@zendavox
```

or from a terminal:

```
claude plugin marketplace add tellstevenj/zendavox-plugin
claude plugin install zendavox-dev@zendavox
```

**3. Start a new session.** It opens with your project's brief. To check the
setup at any time, run `python -m zendavox.dev check` with this plugin's
`vendor` folder on `PYTHONPATH`.

It needs Python 3.10 or newer on your PATH and nothing else - no packages to
install.

## Turning the topic check off

Every alert says how. Set `ZENDAVOX_DRIFT=off` in your environment, or put
`"drift": "off"` in `~/.zendavox/dev.json`. Use `warn` instead of `off` to keep
the alerts without ever holding a message back.

The stop only happens when a model confirms the chat has changed subject.
It asks Claude Code (signed in with `claude` on its own) or, failing that,
the Anthropic API when `ANTHROPIC_API_KEY` is set. With neither, you get the
warning and the chat is never stopped. Checking your messages happens on your
machine; only that one confirming question - the chat's opening and its
latest messages - is sent to Claude.

## Windows first

`bin/zendavox-dev.cmd` is the tested path. `bin/zendavox-dev.sh` is included
for macOS and Linux but has not been run against a live project the way the
Windows one has.

## How it is built

Everything under `vendor/` is plain Python with no third-party imports, so
any Python on the machine runs it. Nothing a hook does can break a session:
every hook exits cleanly whatever happens, and a check that cannot run says
nothing rather than blocking you.

This repository is published from the Zendavox product. Report problems at
<https://www.zendavox.com>.
