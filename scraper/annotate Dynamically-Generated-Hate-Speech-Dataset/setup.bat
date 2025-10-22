@echo off
title Hate Speech Annotation System - Quick Setup
echo Hate Speech Annotation System - Quick Setup
echo ==============================================
echo.

:: --- Check if Python is installed ---
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo Python not found. Please install Python 3.7+ and add it to PATH.
    pause
    exit /b 1
)
echo ✓ Python found

:: --- Install dependencies ---
echo.
echo Installing dependencies...
python -m pip install pandas openai tqdm -q
if %errorlevel% neq 0 (
    echo Failed to install dependencies.
    echo Try manually running: python -m pip install pandas openai tqdm
    pause
    exit /b 1
)
echo ✓ Dependencies installed

:: --- Check for API key ---
echo.
if "%OPENROUTER_API_KEY%"=="" (
    echo OPENROUTER_API_KEY not set.
    echo.
    echo Get your free API key from: https://openrouter.ai
    echo.
    set /p response="Do you want to enter your API key now? (y/n): "
    if /i "%response%"=="y" (
        set /p api_key="Enter your OpenRouter API key: "
        setx OPENROUTER_API_KEY "%api_key%" >nul
        set OPENROUTER_API_KEY=%api_key%
        echo ✓ API key set for this session and saved permanently using setx.
        echo (You may need to reopen the terminal for it to take effect.)
    ) else (
        echo Skipping API key setup.
        echo You can manually set it later with:
        echo     setx OPENROUTER_API_KEY your-api-key-here
    )
) else (
    echo ✓ API key found.
)

:: --- Check for input file ---
echo.
echo Looking for input CSV file...
if exist "input_dataset.csv" (
    echo ✓ Found: input_dataset.csv
    for /f %%A in ('find /c /v "" ^< "input_dataset.csv"') do set rows=%%A
    set /a data_rows=%rows%-1
    echo   Rows: %data_rows%
) else (
    echo No input_dataset.csv found.
    echo.
    echo Please:
    echo 1. Place your CSV file in this directory
    echo 2. Rename it to 'input_dataset.csv', OR
    echo 3. Edit hate_speech_annotator.py line 382 to use your filename
)

echo.
echo ==============================================
echo Setup complete!
echo.
echo To run:
echo   1. Test run (100 rows):  python hate_speech_annotator.py
echo   2. Full dataset:         Edit SAMPLE_SIZE=None in script, then run
echo.
echo Files will be created:
echo   - annotated_dataset.csv (final output)
echo   - annotated_dataset_checkpoint.csv (progress save)
echo   - annotation_summary.txt (statistics)
echo.
pause
