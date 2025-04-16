@echo off
echo Running Traffic Analysis...
call conda activate env
set KMP_DUPLICATE_LIB_OK=TRUE
python main.py
pause 