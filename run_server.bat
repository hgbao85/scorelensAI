@echo off
echo 🎱 Pool8 AI Backend Server
echo ========================
echo.

echo 📦 Activating virtual environment...
call venv\Scripts\activate.bat

echo 🚀 Starting backend server...
python backend_server.py

echo.
echo ⏹️  Server stopped. Press any key to exit...
pause > nul
