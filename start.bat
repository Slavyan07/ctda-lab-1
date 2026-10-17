@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" goto missing
".venv\Scripts\python.exe" -m loliland_agent %*
exit /b %errorlevel%
:missing
echo Run install.bat first.
exit /b 1
