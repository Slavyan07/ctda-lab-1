@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" goto missing
set QT_QPA_PLATFORM=offscreen
".venv\Scripts\python.exe" -m pip check
if errorlevel 1 exit /b 1
".venv\Scripts\python.exe" -m ruff check .
if errorlevel 1 exit /b 1
".venv\Scripts\python.exe" -m ruff format --check .
if errorlevel 1 exit /b 1
".venv\Scripts\python.exe" -m mypy src
if errorlevel 1 exit /b 1
".venv\Scripts\python.exe" -m pytest -q
if errorlevel 1 exit /b 1
echo Copilot checks passed. See WINDOWS_SETUP.md for real screen and voice checks.
exit /b 0
:missing
echo Run install.bat first.
exit /b 1
