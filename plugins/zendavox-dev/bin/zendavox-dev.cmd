@echo off
rem Windows entry point for the zendavox-dev plugin.
rem
rem Runs the vendored, stdlib-only copy of zendavox.dev under vendor/ - not
rem the project's own src/zendavox, so this works in any repo the plugin is
rem installed into, with no venv or install step required.
rem
rem   zendavox-dev.cmd start   session-start hook: fetch the project's brief
rem   zendavox-dev.cmd end     session-end hook: close an open session record
rem   zendavox-dev.cmd prompt  prompt hook: warn when a chat drifts off topic
rem   zendavox-dev.cmd mcp     the MCP server, on stdin and stdout
setlocal

rem %~dp0 is this script's own directory (...\bin\), so ..\ is the plugin root
rem regardless of where the plugin was installed or what invoked this script.
set "PLUGIN_ROOT=%~dp0.."

set "PY=python"
where python >nul 2>nul
if errorlevel 1 (
    where py >nul 2>nul
    if not errorlevel 1 set "PY=py"
)

set "PYTHONPATH=%PLUGIN_ROOT%\vendor;%PYTHONPATH%"

if "%~1"=="mcp" (
    "%PY%" -m zendavox.dev mcp
    exit /b %ERRORLEVEL%
)
if "%~1"=="start" (
    "%PY%" -m zendavox.dev hook start
    rem A hook must never take a session down with it.
    exit /b 0
)
if "%~1"=="end" (
    "%PY%" -m zendavox.dev hook end
    exit /b 0
)
if "%~1"=="prompt" (
    "%PY%" -m zendavox.dev hook prompt
    exit /b 0
)

echo [zendavox-dev] usage: zendavox-dev.cmd {start^|end^|prompt^|mcp} 1>&2
exit /b 0
