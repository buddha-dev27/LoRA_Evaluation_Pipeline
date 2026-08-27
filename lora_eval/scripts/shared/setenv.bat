@echo off
:: ============================================================
:: Environment Setup
:: Edit PYTHON_DIR below to match your system before first use
:: ============================================================

:: Path to the folder containing python.exe
:: Virtual environment (recommended — keeps validator dependencies
:: separate from ComfyUI and AI-Toolkit):
set PYTHON_DIR=G:\output\lora_eval\scripts\venv\Scripts

:: ComfyUI embedded Python (alternative if not using venv):
:: set PYTHON_DIR=G:\ComfyUI-Easy-Install\python_embeded

:: System Python (alternative):
:: set PYTHON_DIR=C:\Python312

:: ============================================================
set PYTHON=%PYTHON_DIR%\python.exe
set PATH=%PYTHON_DIR%;%PATH%

if not exist "%PYTHON%" (
    echo.
    echo ERROR: python.exe not found at %PYTHON%
    echo Edit PYTHON_DIR in setenv.bat to match your system.
    echo.
    pause
    exit /b 1
)
