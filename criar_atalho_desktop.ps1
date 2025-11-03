# Script para criar atalho na area de trabalho
# Executa: CriarAtalhoDesktop.ps1

# Caminhos
$workspacePath = Split-Path -Parent $MyInvocation.MyCommand.Path
$iniciarBatPath = Join-Path $workspacePath "iniciar_cursor_e_app.bat"
$desktopPath = [Environment]::GetFolderPath("Desktop")
$shortcutPath = Join-Path $desktopPath "YOLO Detection System.lnk"

Write-Host "Criando atalho na area de trabalho..." -ForegroundColor Green
Write-Host "Caminho do workspace: $workspacePath" -ForegroundColor Cyan
Write-Host "Arquivo de destino: $iniciarBatPath" -ForegroundColor Cyan

# Verificar se iniciar_cursor_e_app.bat existe
if (-not (Test-Path $iniciarBatPath)) {
    Write-Host "ERRO: Arquivo iniciar_cursor_e_app.bat nao encontrado em: $iniciarBatPath" -ForegroundColor Red
    exit 1
}

# Criar objeto WScript.Shell
$WScriptShell = New-Object -ComObject WScript.Shell

# Criar atalho
$Shortcut = $WScriptShell.CreateShortcut($shortcutPath)
$Shortcut.TargetPath = $iniciarBatPath
$Shortcut.WorkingDirectory = $workspacePath
$Shortcut.Description = "YOLO Detection System - Abre Cursor e executa app.py"
$Shortcut.IconLocation = "python.exe,0"

# Salvar atalho
$Shortcut.Save()

Write-Host "" -ForegroundColor Green
Write-Host "Atalho criado com sucesso!" -ForegroundColor Green
Write-Host "Localizacao: $shortcutPath" -ForegroundColor Cyan
Write-Host "" -ForegroundColor Yellow
Write-Host "Voce pode encontrar o icone 'YOLO Detection System' na sua area de trabalho." -ForegroundColor Yellow
Write-Host "De um duplo clique nele para:" -ForegroundColor Yellow
Write-Host "  1. Abrir o Cursor no diretorio do projeto" -ForegroundColor Cyan
Write-Host "  2. Executar a aplicacao app.py" -ForegroundColor Cyan

# Limpar objeto COM
[System.Runtime.Interopservices.Marshal]::ReleaseComObject($WScriptShell) | Out-Null

