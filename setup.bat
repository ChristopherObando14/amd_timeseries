@echo off
python -m pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
pip install -e . --quiet
mkdir outputs\figures outputs\models outputs\results 2>nul
copy .env.example .env 2>nul
echo.
echo Listo. Ahora corre: python main.py