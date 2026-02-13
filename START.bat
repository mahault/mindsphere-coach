@echo off
title MindSphere Coach
color 0A

echo.
echo  ============================================================
echo    MindSphere Coach — Interactive ToM Coaching Agent
echo  ============================================================
echo.

:: Save the directory this bat file lives in
cd /d "%~dp0"

:: ── Check for Python ──────────────────────────────────────────
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo  [ERROR] Python is not installed or not in your PATH.
    echo.
    echo  Please install Python 3.10+ from https://www.python.org/downloads/
    echo  Make sure to check "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)

:: Show Python version
echo  Found Python:
python --version
echo.

:: ── Install dependencies if needed ────────────────────────────
echo  Checking dependencies...
python -c "import fastapi" >nul 2>&1
if %errorlevel% neq 0 (
    echo  Installing dependencies (first time only, this may take a minute)...
    echo.
    pip install -e . >nul 2>&1
    if %errorlevel% neq 0 (
        echo  [ERROR] Failed to install dependencies.
        echo  Try running manually: pip install -e .
        echo.
        pause
        exit /b 1
    )
    echo  Dependencies installed successfully.
    echo.
)

:: ── Check for .env file ───────────────────────────────────────
if not exist ".env" (
    echo  --------------------------------------------------------
    echo  NOTE: No .env file found.
    echo.
    echo  The app will work without it, but LLM-powered responses
    echo  require a Mistral API key. To enable them:
    echo    1. Copy .env.example to .env
    echo    2. Add your key: MISTRAL_API_KEY=your_key_here
    echo  --------------------------------------------------------
    echo.
)

:: ── Start the server ──────────────────────────────────────────
echo  Starting MindSphere Coach...
echo.
echo  ============================================================
echo    Open your browser to:  http://localhost:8000
echo  ============================================================
echo.
echo  Press Ctrl+C to stop the server.
echo.

:: Open browser automatically after a short delay
start "" "http://localhost:8000"

:: Run the server
python scripts/run_demo.py

:: If we get here, server was stopped
echo.
echo  Server stopped. You can close this window.
pause
