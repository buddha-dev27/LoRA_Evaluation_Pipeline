@echo off
setlocal enabledelayedexpansion
call "%~dp0setenv.bat"

:: ============================================================
:: LoRA Checkpoint Batch Validator
:: Automatically scans the checkpoint folder - no manual list needed.
:: ============================================================

:: --- CONFIGURE THESE ----------------------------------------
:: Model type: krea, minimax, wan, or zit
set MODEL=krea

:: Character to test
set CHARACTER=k41t1yn
:: ------------------------------------------------------------

set ROOT_DIR=G:\output\lora_eval
set SCRIPTS_DIR=%~dp0
set SCRIPT=%SCRIPTS_DIR%lora_validator.py
set COMPARE=%SCRIPTS_DIR%build_comparison.py

set BASE_DIR=%ROOT_DIR%\%MODEL%\%CHARACTER%
set CHECKPOINT_DIR=%BASE_DIR%\checkpoint
set BASELINE_DIR=%BASE_DIR%\baseline
set RESULTS_DIR=%BASE_DIR%\results
if not exist "%RESULTS_DIR%" mkdir "%RESULTS_DIR%"

echo.
echo LoRA Checkpoint Batch Validator
echo ================================
echo Model      : %MODEL%
echo Character  : %CHARACTER%
echo Checkpoints: %CHECKPOINT_DIR%
echo Baseline   : %BASELINE_DIR%
echo Results    : %RESULTS_DIR%
echo.

if not exist "%CHECKPOINT_DIR%" (
    echo ERROR: Checkpoint folder not found: %CHECKPOINT_DIR%
    pause
    exit /b 1
)

if not exist "%BASELINE_DIR%" (
    echo ERROR: Baseline folder not found: %BASELINE_DIR%
    echo Run checkpoint_queue_zit.py or model equivalent first to generate baseline images.
    pause
    exit /b 1
)

:: Count checkpoint folders
set TOTAL=0
for /d %%d in (%CHECKPOINT_DIR%\*) do set /a TOTAL+=1
echo Found %TOTAL% checkpoint folders.
echo.

:: Score each checkpoint folder in sorted order
set IDX=0
for /d %%d in (%CHECKPOINT_DIR%\*) do (
    set /a IDX+=1
    set STEP=%%~nxd
    set LORA_DIR=%%d
    set OUT_CSV=%RESULTS_DIR%\checkpoint_!STEP!_scores.csv

    echo [!IDX!/%TOTAL%] Scoring checkpoint !STEP!...
    "%PYTHON%" "%SCRIPT%" ^
        --dataset   "%BASE_DIR%\dataset" ^
        --baseline  "%BASELINE_DIR%" ^
        --lora      "!LORA_DIR!" ^
        --output    "!OUT_CSV!"

    if errorlevel 1 (
        echo [ERROR] !STEP! failed. Check output above.
    ) else (
        echo [DONE]  !STEP! scored.
    )
    echo.
)

echo Building checkpoint comparison summary...
"%PYTHON%" "%COMPARE%" --character "%CHARACTER%" --mode checkpoint ^
    --root "%ROOT_DIR%\%MODEL%"

echo.
echo All done. Results in %RESULTS_DIR%
pause
