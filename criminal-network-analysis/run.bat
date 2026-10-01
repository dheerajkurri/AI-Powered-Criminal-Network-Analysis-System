@echo off
cd /d "%~dp0"
pip install -r requirements.txt -q
python -m streamlit run app.py
