@echo off
setlocal
cd /d "%~dp0"

if not exist "venv\Scripts\python.exe" (
  echo Venv nao encontrado. Crie com: python -m venv venv
  exit /b 1
)

venv\Scripts\python.exe src\main.py --debug --save-frames --dry-run
