@echo off
chcp 65001 >nul
title llama-server on :11433
rem Edit these three paths, then double-click this file.
set "LLAMA=D:\llama.cpp\b11756\llama-server.exe"
set "MODEL=D:\ollama-gguf\qwen3.8-27b\Qwen3.8-27B-UD-Q3_K_XL.gguf"
set "ALIAS=qwen3.8-27b-q3xl"

"%LLAMA%" -m "%MODEL%" -a "%ALIAS%" ^
  --host 0.0.0.0 --port 11433 ^
  -c 49152 -ngl 99 -np 1 -fa on -ctk q5_1 -ctv q5_1 --jinja

echo.
echo [server stopped]
pause
