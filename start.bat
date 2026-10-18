@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" goto missing
".venv\Scripts\python.exe" -m loliland_agent %*
if errorlevel 1 goto failed
exit /b 0
:missing
echo Сначала запустите install.bat.
:failed
echo Copilot не запустился. Проверьте сообщение выше и выполните check.bat.
pause
exit /b 1
