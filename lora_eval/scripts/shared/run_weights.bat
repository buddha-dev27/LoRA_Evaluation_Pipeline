@echo off
setlocal enabledelayedexpansion
call "%~dp0setenv.bat"

:: ============================================================
:: LoRA Weight Batch Validator
:: ============================================================

:: --- CONFIGURE THESE ----------------------------------------
:: Model type: krea, minimax, wan, or zit
set MODEL=krea

:: Character to test
set CHARACTER=k41t1yn

:: Checkpoint step number being tested (used in folder path and chart title)
set CHECKPOINT_STEP=2700

:: Strength values to score (must match folder names exactly)
set STRENGTHS=0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95 1.00

:: LoRA filename being tested (used in chart title)
set LORA_FILE=%CHARACTER%_%MODEL%_%CHECKPOINT_STEP%.safetensors
:: ------------------------------------------------------------

set ROOT_DIR=G:\output\lora_eval
set SCRIPTS_DIR=%~dp0
set SCRIPT=%SCRIPTS_DIR%lora_validator.py
set COMPARE=%SCRIPTS_DIR%build_comparison.py

set BASE_DIR=%ROOT_DIR%\%MODEL%\%CHARACTER%
set BASELINE_DIR=%BASE_DIR%\baseline
set WEIGHT_DIR=%BASE_DIR%\weight%CHECKPOINT_STEP%
set RESULTS_DIR=%BASE_DIR%\results
if not exist "%RESULTS_DIR%" mkdir "%RESULTS_DIR%"

echo.
echo LoRA Weight Batch Validator
echo ============================
echo Model      : %MODEL%
echo Character  : %CHARACTER%
echo Checkpoint : %CHECKPOINT_STEP% steps
echo Weight dir : %WEIGHT_DIR%
echo Baseline   : %BASELINE_DIR%
echo Results    : %RESULTS_DIR%
echo.

if not exist "%BASELINE_DIR%" (
    echo ERROR: Baseline folder not found: %BASELINE_DIR%
    echo Run checkpoint_queue_%MODEL%.py or model equivalent first to generate baseline images.
    pause
    exit /b 1
)

set TOTAL=0
for %%s in (%STRENGTHS%) do set /a TOTAL+=1

set IDX=0
for %%s in (%STRENGTHS%) do (
    set /a IDX+=1
    set STRENGTH=%%s
    set LORA_DIR=%WEIGHT_DIR%\!STRENGTH!
    set OUT_CSV=%RESULTS_DIR%\weight%CHECKPOINT_STEP%_!STRENGTH!_scores.csv

    if not exist "!LORA_DIR!" (
        echo [SKIP] strength !STRENGTH! - folder not found.
    ) else (
        echo [!IDX!/%TOTAL%] Scoring weight !STRENGTH!...
        "%PYTHON%" "%SCRIPT%" ^
            --dataset   "%BASE_DIR%\dataset" ^
            --baseline  "%BASELINE_DIR%" ^
            --lora      "!LORA_DIR!" ^
            --output    "!OUT_CSV!"

        if errorlevel 1 (
            echo [ERROR] weight !STRENGTH! failed.
        ) else (
            echo [DONE]  weight !STRENGTH! scored.
        )
    )
    echo.
)

echo Building weight comparison summary...
"%PYTHON%" "%COMPARE%" --character "%CHARACTER%" --mode weight ^
    --root "%ROOT_DIR%\%MODEL%" --lora "%LORA_FILE%" ^
    --checkpoint-step "%CHECKPOINT_STEP%"

echo.
echo All done. Results in %RESULTS_DIR%
pause
