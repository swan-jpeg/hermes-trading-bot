@echo off
REM Hermes Trading Bot — Windows installatie-script
REM Vereist: Python 3.11+ en uv (https://docs.astral.sh/uv/)
echo.
echo === Hermes Trading Bot — Windows installatie ===
echo.

REM 1. Controleer uv
where uv >nul 2>nul
if errorlevel 1 (
    echo [FOUT] uv niet gevonden. Installeer eerst: pip install uv
    exit /b 1
)

REM 2. Installeer dependencies (core + data + ml)
echo [1/4] Core + data + ML dependencies installeren...
uv sync --extra dev --extra ml --extra data
if errorlevel 1 exit /b 1

REM 3. Zware multimodale modellen (optioneel — groot)
echo [2/4] Multimodale modellen installeren (whisper, DeepFace, FinBERT)...
uv sync --extra multimodal
if errorlevel 1 exit /b 1

REM 4. MediaPipe (pose) — werkt doorgaans goed op Windows
echo [3/4] MediaPipe installeren (pose/gezichtsdetectie)...
uv sync --extra video
if errorlevel 1 exit /b 1

REM 5. Tests draaien
echo [4/4] Tests draaien...
uv run pytest
if errorlevel 1 exit /b 1

echo.
echo === Installatie klaar! ===
echo.
echo Start de demo:   uv run python -m hermes_bot.demo
echo Start de web-UI: uv run python -m hermes_bot.webui  (http://localhost:9124)
echo.
pause
