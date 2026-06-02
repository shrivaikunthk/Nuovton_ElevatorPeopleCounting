@echo off
REM ===========================================================================
REM compile_vela.bat -- Windows equivalent of scripts/compile_vela.sh
REM
REM Compiles an INT8 TFLite model with Ethos-U Vela for the Nuvoton M55M1
REM (Ethos-U55-256 NPU). The output file is what you copy to the SD card.
REM
REM USAGE:
REM     compile_vela.bat <input_int8.tflite>
REM     compile_vela.bat <input_int8.tflite> <output_dir>
REM
REM EXAMPLE (re-running on a freshly-trained model):
REM     compile_vela.bat keras_fomo\runs\fomo\model_int8.tflite
REM
REM REQUIREMENTS:
REM     * Python 3.10+ on PATH
REM     * Run once: python -m pip install ethos-u-vela==3.10.0
REM
REM You normally do NOT need to run this on Windows. The Vela-compiled
REM model is already shipped in sd_card\model_int8_vela.tflite.
REM ===========================================================================

setlocal enabledelayedexpansion

if "%~1"=="" (
    echo USAGE: compile_vela.bat ^<input_int8.tflite^> [output_dir]
    exit /b 1
)

set INPUT=%~1
set OUTPUT_DIR=%~2
if "%OUTPUT_DIR%"=="" set OUTPUT_DIR=%~dp1vela

if not exist "%INPUT%" (
    echo ERROR: input model not found: %INPUT%
    exit /b 1
)

if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

REM Install vela on demand
python -c "import ethosu.vela" 2>nul
if errorlevel 1 (
    echo [vela] installing ethos-u-vela 3.10.0 ...
    python -m pip install --quiet ethos-u-vela==3.10.0
)

REM Vela config that ships with the Nuvoton YOLO sample
set VELA_INI=%~dp0..\ML_YOLO\yolov8_ultralytics\vela\Tool\vela\default_vela.ini

if not exist "%VELA_INI%" (
    echo WARNING: vela ini not found at %VELA_INI%
    echo          falling back to vela's built-in defaults
    set VELA_INI_ARG=
) else (
    set VELA_INI_ARG=--config "%VELA_INI%"
)

echo [vela] compiling %INPUT% -^> %OUTPUT_DIR%
python -m ethosu.vela ^
    "%INPUT%" ^
    --accelerator-config=ethos-u55-256 ^
    --optimise Size ^
    %VELA_INI_ARG% ^
    --memory-mode=Shared_Sram ^
    --system-config=Ethos_U55_High_End_Embedded ^
    --output-dir="%OUTPUT_DIR%"

if errorlevel 1 (
    echo [vela] FAILED
    exit /b 2
)

for %%F in ("%INPUT%") do set BASENAME=%%~nF
set OUT_FILE=%OUTPUT_DIR%\%BASENAME%_vela.tflite

if exist "%OUT_FILE%" (
    echo [vela] OK  -^> %OUT_FILE%
) else (
    echo [vela] FAILED: expected output file not produced
    exit /b 2
)

endlocal
