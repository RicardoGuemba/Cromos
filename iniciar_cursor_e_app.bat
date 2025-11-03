@echo off
REM ============================================================
REM YOLO Detection System - Iniciador Principal
REM ============================================================
REM Abre o Cursor no diretório do projeto e executa app.py
REM
REM Uso: Duplo clique no atalho da área de trabalho
REM ============================================================

cd /d "%~dp0"

REM ============================================================
REM ETAPA 1: Detectar e abrir Cursor no diretório do projeto
REM ============================================================
where cursor >nul 2>&1
if %errorlevel%==0 (
    REM Cursor encontrado no PATH
    start "" cursor "%CD%"
) else if exist "%LOCALAPPDATA%\Programs\Cursor\Cursor.exe" (
    REM Cursor em LocalAppData
    start "" "%LOCALAPPDATA%\Programs\Cursor\Cursor.exe" "%CD%"
) else if exist "%USERPROFILE%\AppData\Local\Programs\Cursor\Cursor.exe" (
    REM Cursor em AppData do usuário
    start "" "%USERPROFILE%\AppData\Local\Programs\Cursor\Cursor.exe" "%CD%"
) else if exist "C:\Program Files\Cursor\Cursor.exe" (
    REM Cursor em Program Files
    start "" "C:\Program Files\Cursor\Cursor.exe" "%CD%"
) else (
    REM Tentativa final usando comando cursor
    start "" cmd /c "cd /d %CD% && cursor %CD%"
)

REM Aguardar Cursor inicializar
timeout /t 2 /nobreak >nul

REM ============================================================
REM ETAPA 2: Executar app.py em nova janela CMD
REM ============================================================
if exist "executar_app.bat" (
    start "" /D "%CD%" cmd /k "executar_app.bat"
) else (
    REM Fallback: executar inline se script auxiliar não existir
    start "" /D "%CD%" cmd /k "call venv\Scripts\activate.bat 2>nul && set KMP_DUPLICATE_LIB_OK=TRUE && set CUDA_VISIBLE_DEVICES=0 && python app.py || pause"
)
