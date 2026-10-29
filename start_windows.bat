@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Error: .venv\Scripts\python.exe not found!
  pause
  exit /b 1
)
echo ============================================================
echo   FAMILY AI GAME DAY -- LIVE AI REFEREE (ADVANCED TRACK)
echo   Abu Dhabi University Challenge
echo ============================================================
echo   Starting server for PC and Mobile Phone access...
echo   (Press Ctrl+C to stop)
echo ============================================================
".venv\Scripts\python.exe" serve.py %*
pause
