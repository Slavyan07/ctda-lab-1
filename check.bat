@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
set "COPILOT_AUDIO_STATUS=0"
if not exist ".venv\Scripts\python.exe" goto failed
".venv\Scripts\python.exe" -m pip check
if errorlevel 1 goto failed
if /i "%~1"=="--ci" goto ci_devices
".venv\Scripts\python.exe" -m loliland_agent.diagnostics
if errorlevel 2 goto audio_notice
if errorlevel 1 goto failed
goto tests
:audio_notice
set "COPILOT_AUDIO_STATUS=2"
echo Есть замечания по аудиоустройствам. Текстовый режим доступен; проверьте раздел Голосовой напарник.
goto tests
:ci_devices
".venv\Scripts\python.exe" -m loliland_agent.diagnostics --no-audio
if errorlevel 1 goto failed
:tests
set QT_QPA_PLATFORM=offscreen
".venv\Scripts\python.exe" -m ruff check .
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m ruff format --check .
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m mypy src
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pytest -q
if errorlevel 1 goto failed
echo Проверки кода пройдены. Для проверки настоящего микрофона и голоса используйте start.bat.
if /i not "%~1"=="--ci" pause
exit /b %COPILOT_AUDIO_STATUS%
:failed
echo Проверка не пройдена. Выполните install.bat и посмотрите сообщение об ошибке выше.
if /i not "%~1"=="--ci" pause
exit /b 1
