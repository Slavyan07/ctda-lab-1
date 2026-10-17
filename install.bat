@echo off
setlocal
cd /d "%~dp0"
py -3.11 -c "import sys; assert sys.maxsize > 2**32"
if errorlevel 1 goto python_missing
py -3.11 -m venv .venv
if errorlevel 1 exit /b 1
".venv\Scripts\python.exe" -m pip install --require-hashes -r requirements-lock.txt
if errorlevel 1 exit /b 1
".venv\Scripts\python.exe" -m pip install --no-deps -e .
if errorlevel 1 exit /b 1
".venv\Scripts\python.exe" -m pip check
if errorlevel 1 exit /b 1
if not exist ".env" copy ".env.example" ".env" >nul
echo Installed. Run start.bat --demo, then read WINDOWS_SETUP.md.
exit /b 0
:python_missing
echo Install Python 3.11 x64 with the Python Launcher from python.org.
exit /b 1
