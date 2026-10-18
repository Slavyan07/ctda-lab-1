@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
where ollama >nul 2>&1
if errorlevel 1 goto missing
set "OLLAMA_HOST=127.0.0.1:11434"
set "OLLAMA_NUM_PARALLEL=1"
set "OLLAMA_MAX_LOADED_MODELS=1"
set "OLLAMA_GPU_OVERHEAD=4294967296"
set "OLLAMA_CONTEXT_LENGTH=4096"
set "OLLAMA_KEEP_ALIVE=0"
echo Локальный сервер с одной моделью и резервом 4 GB VRAM для игры.
echo Если порт занят, завершите Ollama через его значок вручную и повторите.
echo Copilot не останавливает чужие процессы. Закрытие этого окна остановит созданный сервер.
ollama serve
set "COPILOT_SERVER_RESULT=%ERRORLEVEL%"
if not "%COPILOT_SERVER_RESULT%"=="0" pause
exit /b %COPILOT_SERVER_RESULT%
:missing
echo Установите бесплатный Ollama с https://ollama.com/download/windows и повторите.
pause
exit /b 1
