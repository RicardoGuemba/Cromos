@echo off
REM ============================================================
REM YOLO Detection System - Execução Simples
REM ============================================================
REM Executa app.py sem abrir o Cursor
REM Alternativa ao iniciar_cursor_e_app.bat
REM ============================================================

cd /d "%~dp0"

echo ========================================
echo YOLO Detection System - Basler USB3 Vision
echo ========================================
echo.
echo Diretorio de trabalho: %CD%
echo.

REM Ativar ambiente virtual se existir
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

REM Configurar variáveis de ambiente
set KMP_DUPLICATE_LIB_OK=TRUE
set CUDA_VISIBLE_DEVICES=0

REM Verificar instalação (opcional - pode ser comentado se causar problemas)
REM echo.
REM echo Verificando instalacao rapida...
REM python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA disponivel:', torch.cuda.is_available())" 2>nul
REM if errorlevel 1 (
REM     echo [AVISO] Nao foi possivel verificar CUDA, mas continuando...
REM )

echo.
echo Iniciando aplicacao...
echo.

REM Executar aplicação
python app.py

REM Manter janela aberta em caso de erro para ver a mensagem
if errorlevel 1 (
    echo.
    echo ========================================
    echo ERRO ao executar aplicacao!
    echo ========================================
    echo.
    echo Verifique os logs em: logs\
    echo.
    pause
)

