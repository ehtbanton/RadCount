@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"
title RadCount Setup

echo.
echo  ============================================================
echo   RadCount - Automated Setup
echo  ============================================================

REM ── Find Python ─────────────────────────────────────────────
echo.
echo  [*] Checking for Python...

set "PY="

REM 1. py launcher (installed to C:\Windows, always in PATH)
py -3 --version >nul 2>&1
if !errorlevel! equ 0 (
    set "PY=py -3"
    goto :python_ok
)

REM 2. python command (verify it's real, not the MS Store alias)
python -c "import sys" >nul 2>&1
if !errorlevel! equ 0 (
    set "PY=python"
    goto :python_ok
)

REM 3. python3 (uncommon on Windows but possible)
python3 -c "import sys" >nul 2>&1
if !errorlevel! equ 0 (
    set "PY=python3"
    goto :python_ok
)

REM 4. Check common install directories
for %%V in (313 312 311 310) do (
    if exist "%LOCALAPPDATA%\Programs\Python\Python%%V\python.exe" (
        set "PY=%LOCALAPPDATA%\Programs\Python\Python%%V\python.exe"
        goto :python_ok
    )
)
for %%V in (313 312 311 310) do (
    if exist "C:\Python%%V\python.exe" (
        set "PY=C:\Python%%V\python.exe"
        goto :python_ok
    )
)

REM ── Python not found — install it ───────────────────────────
echo  [!] Python not found. Attempting to install...
echo.

REM Try winget first (built into Windows 11 and Windows 10 1709+)
winget --version >nul 2>&1
if !errorlevel! equ 0 (
    echo  [*] Installing Python 3.12 via Windows Package Manager...
    winget install Python.Python.3.12 --accept-package-agreements --accept-source-agreements
    echo.

    REM py launcher is installed to C:\Windows — available immediately
    py -3 --version >nul 2>&1
    if !errorlevel! equ 0 (
        set "PY=py -3"
        goto :python_ok
    )
)

REM Fallback: download installer directly
echo  [*] Downloading Python installer from python.org...
powershell -ExecutionPolicy Bypass -Command ^
    "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/3.12.7/python-3.12.7-amd64.exe' -OutFile '%TEMP%\python_setup.exe'"

if not exist "%TEMP%\python_setup.exe" (
    goto :python_fail
)

echo  [*] Installing Python (this may take a minute)...
"%TEMP%\python_setup.exe" /passive InstallAllUsers=0 PrependPath=1 Include_test=0 Include_launcher=1
del "%TEMP%\python_setup.exe" >nul 2>&1
echo.

REM Try py launcher again
py -3 --version >nul 2>&1
if !errorlevel! equ 0 (
    set "PY=py -3"
    goto :python_ok
)

REM Try common paths after fresh install
for %%V in (312 313) do (
    if exist "%LOCALAPPDATA%\Programs\Python\Python%%V\python.exe" (
        set "PY=%LOCALAPPDATA%\Programs\Python\Python%%V\python.exe"
        goto :python_ok
    )
)

:python_fail
echo.
echo  [-] ERROR: Could not install Python automatically.
echo  [-] Please install Python 3.10+ from https://www.python.org/downloads/
echo  [-] IMPORTANT: Check "Add Python to PATH" during installation.
echo  [-] Then run this script again.
goto :done

:python_ok
for /f "tokens=*" %%v in ('!PY! --version 2^>^&1') do echo  [+] %%v

REM ── GPU Check ───────────────────────────────────────────────
echo.
nvidia-smi >nul 2>&1
if !errorlevel! equ 0 (
    for /f "tokens=*" %%g in ('nvidia-smi --query-gpu^=name^,memory.total --format^=csv^,noheader') do (
        echo  [+] GPU: %%g
    )
) else (
    echo  [!] No NVIDIA GPU detected ^(or drivers not installed^).
    echo  [!] LLM inference will use CPU — this is much slower.
)

REM ── Run startup.py ──────────────────────────────────────────
echo.
echo  [*] Launching Python setup...
echo  ============================================================
echo.

!PY! "%~dp0startup.py"

:done
echo.
echo  ============================================================
echo  Press any key to exit...
pause >nul
exit /b
