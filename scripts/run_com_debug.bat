@echo off
REM Script de execucao com debug completo

REM Garantir que estamos no diretorio correto
cd /d "%~dp0"

echo ========================================
echo YOLO Detection System - Modo Debug
echo ========================================
echo.

REM Ativar ambiente virtual se existir
if exist venv\Scripts\activate.bat (
    echo [OK] Ambiente virtual encontrado. Ativando...
    call venv\Scripts\activate.bat
) else (
    echo [ERRO] Ambiente virtual nao encontrado!
    echo Execute a instalacao primeiro.
    pause
    exit /b 1
)

REM Configurar variaveis de ambiente
set KMP_DUPLICATE_LIB_OK=TRUE
set CUDA_VISIBLE_DEVICES=0

echo.
echo Executando diagnostico completo...
echo.

REM Executar diagnostico
python diagnostico.py

if errorlevel 1 (
    echo.
    echo ERRO no diagnostico!
    pause
    exit /b 1
)

echo.
echo.
echo ========================================
echo Agora testando app.py completo...
echo ========================================
echo.
echo Pressione Ctrl+C se travar por mais de 30 segundos...
echo.

REM Executar app.py com timeout visual
python app.py

if errorlevel 1 (
    echo.
    echo ERRO ao executar app.py!
    echo.
    echo Verifique os logs em: logs\
    echo.
    pause
)

