@echo off
REM ============================================================
REM YOLO Detection System - Executor da Aplicação
REM ============================================================
REM Script auxiliar que executa app.py com ambiente configurado
REM Usado internamente por iniciar_cursor_e_app.bat
REM ============================================================

cd /d "%~dp0"
title YOLO Detection System - App.py

echo ========================================
echo YOLO Detection System - Basler USB3 Vision
echo ========================================
echo.
echo Diretorio: %CD%
echo.

REM ============================================================
REM ETAPA 1: Configurar ambiente virtual Python
REM ============================================================
if exist venv\Scripts\activate.bat (
    echo [OK] Ambiente virtual encontrado. Ativando...
    call venv\Scripts\activate.bat
) else (
    echo [INSTALACAO] Ambiente virtual nao encontrado.
    echo [INSTALACAO] Criando ambiente virtual e instalando pacotes...
    echo [INSTALACAO] Isso pode levar varios minutos na primeira vez. Aguarde...
    echo.
    python -m venv venv
    call venv\Scripts\activate.bat
    echo.
    echo [INSTALACAO] Atualizando pip...
    python -m pip install --upgrade pip
    echo.
    echo [INSTALACAO] Instalando PyTorch com CUDA (pode demorar...)... 
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    echo.
    echo [INSTALACAO] Instalando outras dependencias...
    pip install -r requirements.txt
    echo.
    echo [OK] Instalacao concluida! Nas proximas vezes sera mais rapido.
    echo.
)

REM ============================================================
REM ETAPA 2: Configurar variáveis de ambiente
REM ============================================================
set KMP_DUPLICATE_LIB_OK=TRUE
set CUDA_VISIBLE_DEVICES=0

echo.
echo ============================================================
echo Iniciando aplicacao app.py...
echo ============================================================
echo.

REM ============================================================
REM ETAPA 3: Executar aplicação principal
REM ============================================================
python app.py

REM ============================================================
REM ETAPA 4: Tratamento de erros
REM ============================================================
if errorlevel 1 (
    echo.
    echo ========================================
    echo ERRO ao executar aplicacao!
    echo ========================================
    echo.
    echo Verifique os logs em: logs\
    echo.
    pause
) else (
    echo.
    echo Aplicacao finalizada normalmente.
    timeout /t 3 /nobreak >nul
)

