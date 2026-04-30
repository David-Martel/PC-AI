@echo off
REM Force UTF-8 encoding for CMD
chcp 65001 > nul 2>&1
set LANG=en_US.UTF-8
set LC_ALL=en_US.UTF-8
set PYTHONIOENCODING=utf-8
set WSL_UTF8=1