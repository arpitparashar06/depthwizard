@echo off
rem DepthWizard on Windows: double-click. The first run sets up .venv and downloads the model.
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo == first run: creating .venv ==
  py -3 -m venv .venv || python -m venv .venv || (echo Python 3.9+ is required & pause & exit /b 1)
  .venv\Scripts\python -m pip install -r requirements.txt || (pause & exit /b 1)
)
if not defined PORT set PORT=8000
.venv\Scripts\python -c "import urllib.request; urllib.request.urlopen('https://huggingface.co', timeout=3)" >nul 2>&1
if errorlevel 1 (
  set HF_HUB_OFFLINE=1
  set TRANSFORMERS_OFFLINE=1
  echo    no internet - using cached model weights and cached DEMs only
)
echo == DepthWizard on http://127.0.0.1:%PORT%  (close this window to stop) ==
start "" cmd /c "timeout /t 10 >nul & start http://127.0.0.1:%PORT%"
.venv\Scripts\python backend\server.py
pause
