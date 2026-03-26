@echo off
echo Setting up layout_project...
python -m venv venv
call venv\Scripts\activate.bat
pip install -r requirements.txt
echo.
echo Setup complete. Virtual environment is active.
echo Run this to activate later: venv\Scripts\activate.bat
pause
