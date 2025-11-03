# ============================================================================
# INSTALADOR - Sistema de Detecção YOLO Basler USB3 Vision
# ============================================================================
# Este script instala o aplicativo na área de trabalho com ambiente virtual
# isolado e todas as dependências necessárias.
# ============================================================================

param(
    [switch]$Force = $false
)

$ErrorActionPreference = "Stop"

# Configurações
$AppName = "YOLO Detection System"
$DesktopPath = [Environment]::GetFolderPath("Desktop")
$InstallPath = Join-Path $DesktopPath $AppName
$VenvPath = Join-Path $InstallPath ".venv"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 70) -ForegroundColor Cyan
Write-Host "  INSTALADOR - " -NoNewline -ForegroundColor Yellow
Write-Host $AppName -ForegroundColor Green
Write-Host ("=" * 72) -ForegroundColor Cyan
Write-Host ""

# ============================================================================
# VERIFICAÇÕES INICIAIS
# ============================================================================

Write-Host "[1/8] Verificando pré-requisitos..." -ForegroundColor Yellow

# Verificar Python
Write-Host "  Verificando Python..." -NoNewline
try {
    $pythonVersion = python --version 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "Python não encontrado"
    }
    Write-Host " ✓ " -ForegroundColor Green -NoNewline
    Write-Host $pythonVersion
    
    # Verificar versão mínima (3.8+)
    $versionMatch = $pythonVersion -match "(\d+)\.(\d+)"
    if ($versionMatch) {
        $major = [int]$Matches[1]
        $minor = [int]$Matches[2]
        if ($major -lt 3 -or ($major -eq 3 -and $minor -lt 8)) {
            Write-Host "  ✗ Python 3.8+ é necessário. Versão encontrada: $pythonVersion" -ForegroundColor Red
            exit 1
        }
    }
} catch {
    Write-Host " ✗ " -ForegroundColor Red
    Write-Host "  ERRO: Python não encontrado!" -ForegroundColor Red
    Write-Host "  Instale Python 3.8+ de https://www.python.org/downloads/" -ForegroundColor Yellow
    exit 1
}

# Verificar pip
Write-Host "  Verificando pip..." -NoNewline
try {
    python -m pip --version 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "pip não encontrado"
    }
    Write-Host " ✓ " -ForegroundColor Green
} catch {
    Write-Host " ✗ " -ForegroundColor Red
    Write-Host "  ERRO: pip não encontrado!" -ForegroundColor Red
    Write-Host "  Execute: python -m ensurepip --upgrade" -ForegroundColor Yellow
    exit 1
}

Write-Host ""

# ============================================================================
# VERIFICAR/CRIAR DIRETÓRIO DE INSTALAÇÃO
# ============================================================================

Write-Host "[2/8] Preparando diretório de instalação..." -ForegroundColor Yellow

if (Test-Path $InstallPath) {
    if ($Force) {
        Write-Host "  Removendo instalação anterior..." -ForegroundColor Yellow
        Remove-Item -Path $InstallPath -Recurse -Force -ErrorAction SilentlyContinue
        Start-Sleep -Seconds 1
    } else {
        Write-Host "  ✗ Diretório já existe: $InstallPath" -ForegroundColor Red
        Write-Host "  Use -Force para sobrescrever ou remova manualmente" -ForegroundColor Yellow
        $response = Read-Host "  Deseja continuar mesmo assim? (S/N)"
        if ($response -ne "S" -and $response -ne "s") {
            exit 0
        }
        Remove-Item -Path $InstallPath -Recurse -Force -ErrorAction SilentlyContinue
    }
}

# Criar diretório
New-Item -ItemType Directory -Path $InstallPath -Force | Out-Null
Write-Host "  ✓ Diretório criado: $InstallPath" -ForegroundColor Green
Write-Host ""

# ============================================================================
# COPIAR ARQUIVOS DO PROJETO
# ============================================================================

Write-Host "[3/8] Copiando arquivos do projeto..." -ForegroundColor Yellow

# Lista de arquivos e diretórios essenciais
$ItemsToCopy = @(
    "app.py",
    "camera_basler.py",
    "infer.py",
    "ui_v2.py",
    "config_manager.py",
    "requirements.txt",
    "README.md",
    "config",
    "models"
)

# Copiar arquivos principais
foreach ($item in $ItemsToCopy) {
    $sourcePath = Join-Path $ScriptDir $item
    $destPath = Join-Path $InstallPath $item
    
    if (Test-Path $sourcePath) {
        if (Test-Path $sourcePath -PathType Container) {
            Copy-Item -Path $sourcePath -Destination $destPath -Recurse -Force
            Write-Host "  ✓ Copiado diretório: $item" -ForegroundColor Green
        } else {
            Copy-Item -Path $sourcePath -Destination $destPath -Force
            Write-Host "  ✓ Copiado arquivo: $item" -ForegroundColor Green
        }
    } else {
        Write-Host "  ⚠ Não encontrado: $item" -ForegroundColor Yellow
    }
}

# Copiar scripts auxiliares (se existirem)
$ScriptFiles = @("run.bat", "executar_app.bat")
foreach ($script in $ScriptFiles) {
    $sourcePath = Join-Path $ScriptDir $script
    if (Test-Path $sourcePath) {
        Copy-Item -Path $sourcePath -Destination (Join-Path $InstallPath $script) -Force
        Write-Host "  ✓ Copiado script: $script" -ForegroundColor Green
    }
}

# Copiar documentação PDF (se existir)
$PdfPath = Join-Path $ScriptDir "Documentacao_App.py.pdf"
if (Test-Path $PdfPath) {
    Copy-Item -Path $PdfPath -Destination (Join-Path $InstallPath "Documentacao_App.py.pdf") -Force
    Write-Host "  ✓ Copiado PDF de documentação" -ForegroundColor Green
} else {
    Write-Host "  ⚠ PDF de documentação não encontrado (gerando...)" -ForegroundColor Yellow
    # Tentar gerar PDF se script existir
    $PdfScriptPath = Join-Path $ScriptDir "gerar_documentacao_pdf.py"
    if (Test-Path $PdfScriptPath) {
        Write-Host "  Gerando PDF de documentação..." -NoNewline
        python $PdfScriptPath 2>&1 | Out-Null
        if (Test-Path $PdfPath) {
            Copy-Item -Path $PdfPath -Destination (Join-Path $InstallPath "Documentacao_App.py.pdf") -Force
            Write-Host " ✓" -ForegroundColor Green
        } else {
            Write-Host " ✗ (falha)" -ForegroundColor Yellow
        }
    }
}

Write-Host ""

# ============================================================================
# CRIAR AMBIENTE VIRTUAL
# ============================================================================

Write-Host "[4/8] Criando ambiente virtual Python..." -ForegroundColor Yellow

Push-Location $InstallPath

try {
    # Remover ambiente virtual existente se houver
    if (Test-Path $VenvPath) {
        Write-Host "  Removendo ambiente virtual existente..." -ForegroundColor Yellow
        Remove-Item -Path $VenvPath -Recurse -Force -ErrorAction SilentlyContinue
    }
    
    # Criar novo ambiente virtual
    Write-Host "  Criando ambiente virtual..." -NoNewline
    python -m venv $VenvPath
    
    if ($LASTEXITCODE -ne 0) {
        throw "Falha ao criar ambiente virtual"
    }
    
    Write-Host " ✓" -ForegroundColor Green
    
    # Ativar ambiente virtual
    $ActivateScript = Join-Path $VenvPath "Scripts\Activate.ps1"
    if (-not (Test-Path $ActivateScript)) {
        throw "Script de ativação não encontrado"
    }
    
    Write-Host "  Ambiente virtual criado: $VenvPath" -ForegroundColor Green
    
} catch {
    Write-Host " ✗" -ForegroundColor Red
    Write-Host "  ERRO: $($_.Exception.Message)" -ForegroundColor Red
    Pop-Location
    exit 1
}

Write-Host ""

# ============================================================================
# ATUALIZAR PIP
# ============================================================================

Write-Host "[5/8] Atualizando pip..." -ForegroundColor Yellow

try {
    & "$VenvPath\Scripts\python.exe" -m pip install --upgrade pip --quiet
    Write-Host "  ✓ pip atualizado" -ForegroundColor Green
} catch {
    Write-Host "  ⚠ Aviso ao atualizar pip (continuando...)" -ForegroundColor Yellow
}

Write-Host ""

# ============================================================================
# INSTALAR DEPENDÊNCIAS
# ============================================================================

Write-Host "[6/8] Instalando dependências Python..." -ForegroundColor Yellow
Write-Host "  Isso pode levar alguns minutos..." -ForegroundColor Gray

$RequirementsPath = Join-Path $InstallPath "requirements.txt"

if (-not (Test-Path $RequirementsPath)) {
    Write-Host "  ✗ requirements.txt não encontrado!" -ForegroundColor Red
    Pop-Location
    exit 1
}

try {
    # Instalar PyTorch com CUDA primeiro (se disponível)
    Write-Host "  Instalando PyTorch com CUDA 12.1..." -NoNewline
    & "$VenvPath\Scripts\python.exe" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --quiet
    if ($LASTEXITCODE -eq 0) {
        Write-Host " ✓" -ForegroundColor Green
    } else {
        Write-Host " ⚠ (tentando sem CUDA)" -ForegroundColor Yellow
        & "$VenvPath\Scripts\python.exe" -m pip install torch torchvision --quiet
    }
    
    # Instalar outras dependências
    Write-Host "  Instalando outras dependências..." -NoNewline
    & "$VenvPath\Scripts\python.exe" -m pip install -r $RequirementsPath --quiet
    
    if ($LASTEXITCODE -ne 0) {
        throw "Falha ao instalar dependências"
    }
    
    Write-Host " ✓" -ForegroundColor Green
    
    # Instalar dependência adicional para PDF (reportlab)
    Write-Host "  Instalando reportlab (para documentação PDF)..." -NoNewline
    & "$VenvPath\Scripts\python.exe" -m pip install reportlab --quiet
    Write-Host " ✓" -ForegroundColor Green
    
} catch {
    Write-Host " ✗" -ForegroundColor Red
    Write-Host "  ERRO ao instalar dependências: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "  Tentando instalação sem --quiet para ver erros..." -ForegroundColor Yellow
    
    # Tentar novamente sem --quiet para ver erros
    & "$VenvPath\Scripts\python.exe" -m pip install -r $RequirementsPath
}

Write-Host ""

# ============================================================================
# GERAR SCRIPT DE EXECUÇÃO
# ============================================================================

Write-Host "[7/8] Criando script de execução..." -ForegroundColor Yellow

$LauncherScript = @"
@echo off
REM ============================================================================
REM Launcher - Sistema de Detecção YOLO
REM ============================================================================

cd /d "%~dp0"

REM Ativar ambiente virtual
call .venv\Scripts\activate.bat

REM Configurar variáveis de ambiente
set KMP_DUPLICATE_LIB_OK=TRUE

REM Executar aplicação
python app.py

REM Pausar se houver erro
if errorlevel 1 (
    echo.
    echo [ERRO] Aplicacao finalizada com erros
    pause
)
"@

$LauncherPath = Join-Path $InstallPath "Executar Aplicativo.bat"
$LauncherScript | Out-File -FilePath $LauncherPath -Encoding ASCII -Force
Write-Host "  ✓ Script de execução criado" -ForegroundColor Green

# Criar também script PowerShell
$LauncherPS = @"
# ============================================================================
# Launcher PowerShell - Sistema de Detecção YOLO
# ============================================================================

Set-Location $InstallPath

# Ativar ambiente virtual
& .\.venv\Scripts\Activate.ps1

# Configurar variáveis de ambiente
\$env:KMP_DUPLICATE_LIB_OK = "TRUE"

# Executar aplicação
python app.py

# Aguardar se houver erro
if (\$LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "[ERRO] Aplicacao finalizada com erros" -ForegroundColor Red
    Read-Host "Pressione ENTER para sair"
}
"@

$LauncherPSPath = Join-Path $InstallPath "Executar Aplicativo.ps1"
$LauncherPS | Out-File -FilePath $LauncherPSPath -Encoding UTF8 -Force
Write-Host "  ✓ Script PowerShell criado" -ForegroundColor Green

Write-Host ""

# ============================================================================
# CRIAR ATALHO NA ÁREA DE TRABALHO
# ============================================================================

Write-Host "[8/8] Criando atalho na área de trabalho..." -ForegroundColor Yellow

try {
    $WshShell = New-Object -ComObject WScript.Shell
    $Shortcut = $WshShell.CreateShortcut("$DesktopPath\$AppName.lnk")
    $Shortcut.TargetPath = $LauncherPath
    $Shortcut.WorkingDirectory = $InstallPath
    $Shortcut.Description = "Sistema de Detecção YOLO - Basler USB3 Vision"
    $Shortcut.IconLocation = "$env:SystemRoot\System32\SHELL32.dll,1"  # Ícone de pasta
    $Shortcut.Save()
    
    Write-Host "  ✓ Atalho criado na área de trabalho" -ForegroundColor Green
} catch {
    Write-Host "  ⚠ Não foi possível criar atalho automaticamente" -ForegroundColor Yellow
    Write-Host "    Você pode criar manualmente apontando para:" -ForegroundColor Gray
    Write-Host "    $LauncherPath" -ForegroundColor Gray
}

Write-Host ""

Pop-Location

# ============================================================================
# RESUMO DA INSTALAÇÃO
# ============================================================================

Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 70) -ForegroundColor Cyan
Write-Host "  INSTALAÇÃO CONCLUÍDA COM SUCESSO!" -ForegroundColor Green
Write-Host ("=" * 72) -ForegroundColor Cyan
Write-Host ""
Write-Host "Diretório de instalação:" -ForegroundColor Yellow
Write-Host "  $InstallPath" -ForegroundColor White
Write-Host ""
Write-Host "Como executar:" -ForegroundColor Yellow
Write-Host "  1. Dê duplo clique no atalho '$AppName' na área de trabalho" -ForegroundColor White
Write-Host "  2. Ou execute: $LauncherPath" -ForegroundColor White
Write-Host ""
Write-Host "Documentação:" -ForegroundColor Yellow
Write-Host "  O PDF com documentação detalhada está em:" -ForegroundColor White
Write-Host "  $InstallPath\Documentacao_App.py.pdf" -ForegroundColor Cyan
Write-Host ""
Write-Host "Pressione qualquer tecla para sair..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")

