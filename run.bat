@echo off
title mklgemm-grid-monitor (Intel oneAPI oneMKL cblas_sgemm)
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" "%~dp0main.py"
) else (
    python "%~dp0main.py"
)
if errorlevel 1 pause
