@echo off
REM Hermes Trading Bot — Windows install script
REM Requires: Python 3.11+ and uv (https://docs.astral.sh/uv/)
echo.
echo === Hermes Trading Bot — Windows install ===
echo.

REM 1. Check uv
where uv >nul 2>nul
if errorlevel 1 (
    echo [ERROR] uv not found. Install first: pip install uv
    exit /b 1
)

REM 2. Install dependencies (core + data + ml)
echo [1/4] Installing core + data + ML dependencies...
uv sync --extra dev --extra ml --extra data
if errorlevel 1 exit /b 1

REM 3. Heavy multimodal models (optional — large)
echo [2/4] Installing multimodal models (whisper, DeepFace, FinBERT)...
uv sync --extra multimodal
if errorlevel 1 exit /b 1

REM 4. MediaPipe (pose) — usually works well on Windows
echo [3/4] Installing MediaPipe (pose/face detection)...
uv sync --extra video
if errorlevel 1 exit /b 1

REM 5. Run tests
echo [4/4] Running tests...
uv run pytest
if errorlevel 1 exit /b 1

echo.
echo === Install complete! ===
echo.
echo Start the demo:   uv run python -m hermes_bot.demo
echo Start the web UI: uv run python -m hermes_bot.webui  (http://localhost:9124)
echo.
pause
