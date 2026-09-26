#!/usr/bin/env bash
#
# POSIX entry point for the zendavox-dev plugin (macOS/Linux/WSL).
#
# Runs the vendored, stdlib-only copy of zendavox.dev under vendor/ - not the
# project's own src/zendavox, so this works in any repo the plugin is
# installed into, with no venv or install step required.
#
#   zendavox-dev.sh start   session-start hook: fetch the project's brief
#   zendavox-dev.sh end     session-end hook: close an open session record
#   zendavox-dev.sh prompt  prompt hook: warn when a chat drifts off topic
#   zendavox-dev.sh mcp     the MCP server, on stdin and stdout
#
# Note the missing `set -e`. A hook that fails must not take a session with
# it: the worst this script is allowed to do is exit 0 having done nothing.
set -uo pipefail

command="${1:-}"

plugin_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

python_bin=""
for candidate in python3 python; do
  if command -v "$candidate" >/dev/null 2>&1; then
    python_bin="$candidate"
    break
  fi
done

if [ -z "$python_bin" ]; then
  echo "[zendavox-dev] no Python found; skipping." >&2
  exit 0
fi

export PYTHONPATH="$plugin_root/vendor${PYTHONPATH:+:$PYTHONPATH}"

case "$command" in
  mcp)
    exec "$python_bin" -m zendavox.dev mcp
    ;;
  start|end|prompt)
    "$python_bin" -m zendavox.dev hook "$command"
    ;;
  *)
    echo "[zendavox-dev] usage: zendavox-dev.sh {start|end|prompt|mcp}" >&2
    ;;
esac

exit 0
