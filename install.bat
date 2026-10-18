@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
set "COPILOT_PY=py -3.11"
py -3.11 -c "import sys; assert sys.version_info[:2]==(3,11) and sys.maxsize>2**32" >nul 2>&1
if not errorlevel 1 goto python_ok
set "COPILOT_PY=python"
python -c "import sys; assert sys.version_info[:2]==(3,11) and sys.maxsize>2**32" >nul 2>&1
if not errorlevel 1 goto python_ok
echo Нужен Python 3.11 x64 с python.org. При установке включите Python Launcher и Add to PATH.
goto failed
:python_ok
echo Устанавливаю LoliLand AI Copilot. Нужен Интернет для зависимостей; модели Whisper не скачиваются.
%COPILOT_PY% -m venv .venv
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip install --disable-pip-version-check --require-hashes -r requirements-lock.txt
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip install --disable-pip-version-check --no-deps -e .
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip check
if errorlevel 1 goto failed
if not exist ".env" copy ".env.example" ".env" >nul
echo Установка завершена. Запустите start.bat. Без ключа доступны DEMO и локальные команды.
echo Для облачного ИИ откройте Настройки API; ключ сохраняется в Windows Credential Manager.
if /i not "%~1"=="--ci" pause
exit /b 0
:failed
echo Установка не завершена. Проверьте сообщение выше, Python 3.11 x64, подключение и права на папку.
if /i not "%~1"=="--ci" pause
exit /b 1
