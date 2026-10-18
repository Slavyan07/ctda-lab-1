@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" goto failed
echo Устанавливаю необязательные зависимости faster-whisper для CPU. Может потребоваться несколько сотен MB.
echo Модель не скачивается: разрешение первой загрузки задаётся отдельно в панели голоса.
".venv\Scripts\python.exe" -m pip install --disable-pip-version-check --require-hashes -r requirements-local-lock.txt
if errorlevel 1 goto failed
echo Готово. Выберите local, tiny и разрешите загрузку модели в панели при необходимости.
if /i not "%~1"=="--ci" pause
exit /b 0
:failed
echo Локальное STT не установлено. Сначала выполните install.bat и проверьте сеть.
if /i not "%~1"=="--ci" pause
exit /b 1
