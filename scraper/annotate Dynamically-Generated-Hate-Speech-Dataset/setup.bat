@echo off
title Multithreaded Hate Speech Annotator - Quick Start
chcp 65001 >nul

echo ╔════════════════════════════════════════════════════════════════════╗
echo ║                                                                    ║
echo ║        MULTITHREADED HATE SPEECH ANNOTATOR - QUICK START           ║
echo ║                                                                    ║
echo ╚════════════════════════════════════════════════════════════════════╝
echo.

:: --- Check Python ---
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo ❌ Python not found. Please install Python 3.7+ and add it to PATH.
    pause
    exit /b 1
)
echo ✅ Python found

:: --- Install dependencies ---
echo.
echo 📦 Installing dependencies...
python -m pip install pandas openai tiktoken rich -q
if %errorlevel% neq 0 (
    echo ❌ Failed to install dependencies.
    echo Try running manually: python -m pip install pandas openai tiktoken rich
    pause
    exit /b 1
)
echo ✅ Dependencies installed

:: --- Create config from template ---
echo.
if not exist "config.ini" (
    echo 📝 Creating config.ini from template...
    if exist "config_template.ini" (
        copy "config_template.ini" "config.ini" >nul
        echo ⚠️  Please edit config.ini and add your OpenAI API key!
        echo.
        echo Get your API key from: https://platform.openai.com/api-keys
        echo.
        set /p openfile="Press ENTER to open config.ini for editing..."
        if exist "config.ini" (
            start notepad config.ini
        )
    ) else (
        echo ❌ Template config_template.ini not found!
        pause
        exit /b 1
    )
) else (
    echo ✅ config.ini already exists
)

:: --- Check if API key is set ---
findstr /C:"YOUR_OPENAI_API_KEY_HERE" "config.ini" >nul
if %errorlevel%==0 (
    echo.
    echo ⚠️  API key not set in config.ini
    echo Please edit config.ini and add your OpenAI API key
    pause
    exit /b 1
)

echo.
echo ╔════════════════════════════════════════════════════════════════════╗
echo ║                         SETUP COMPLETE!                            ║
echo ╚════════════════════════════════════════════════════════════════════╝
echo.
echo 🧪 Run tests:
echo     python test_suite.py
echo.
echo 🚀 Run annotation (sample 100 rows):
echo     python hate_speech_annotator_multithreaded.py --sample 100
echo.
echo 🚀 Run full dataset:
echo     python hate_speech_annotator_multithreaded.py --input your_file.csv
echo.
echo 💡 Features:
echo     • 10-15x faster than single-threaded
echo     • Live progress display
echo     • Ctrl+C saves progress (graceful shutdown)
echo     • Auto-resume from checkpoints
echo     • Thread-safe operations
echo.

pause
