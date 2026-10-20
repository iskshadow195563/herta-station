@echo off
chcp 65001 >nul
title stop llama-server :11433
set "PORT=11433"
set "FOUND="
for /f "tokens=5" %%p in ('netstat -ano ^| findstr ":%PORT%" ^| findstr LISTENING') do (
  echo killing pid %%p
  taskkill /F /T /PID %%p >nul 2>&1
  set "FOUND=1"
)
if not defined FOUND echo no listener on port %PORT%
echo VRAM released.
timeout /t 2 >nul
