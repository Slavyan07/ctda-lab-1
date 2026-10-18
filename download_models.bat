@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" goto missing
".venv\Scripts\python.exe" -m loliland_agent.local.models status
echo.
echo 1 - Скачать Qwen3-VL 4b, основная модель ~3.3 GB
 echo 2 - Скачать Qwen3-VL 2b, облегчённая модель ~2 GB
 echo 3 - Скачать русский Piper Irina и MODEL_CARD ~65 MB
 echo 4 - Только статус, без загрузки
choice /c 1234 /n /m "Выберите действие: "
if errorlevel 4 goto done
if errorlevel 3 goto piper
if errorlevel 2 goto light
".venv\Scripts\python.exe" -m loliland_agent.local.models ollama --model qwen3-vl:4b --download
goto result
:light
".venv\Scripts\python.exe" -m loliland_agent.local.models ollama --model qwen3-vl:2b --download
goto result
:piper
".venv\Scripts\python.exe" -m loliland_agent.local.models piper --download
:result
if errorlevel 1 goto failed
:done
echo Whisper tiny/base загружается при первом распознавании только с флажком согласия в панели.
echo При выборе 2b укажите qwen3-vl:2b в разделе Локальные модели.
pause
exit /b 0
:failed
echo Загрузка не завершена. Проверьте сообщение, сеть и место. Платного резерва нет.
pause
exit /b 1
:missing
echo Сначала запустите install.bat.
pause
exit /b 1
