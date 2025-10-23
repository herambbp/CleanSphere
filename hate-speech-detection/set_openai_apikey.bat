@echo off
REM ===============================================
REM Set OpenAI API Key for current CMD session only
REM ===============================================

echo.
set /p OPENAI_API_KEY=Enter your OpenAI API key: 
echo.

REM Confirm the variable was set
echo Your OpenAI API key has been set for this session.
echo To verify, run:
echo     echo %%OPENAI_API_KEY%%
echo.

REM Optional: export for immediate use in Python or Node
REM Example usage:
REM     python your_script.py
REM or  node your_script.js

pause
