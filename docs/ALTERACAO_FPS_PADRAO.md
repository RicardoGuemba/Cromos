# ALTERAÇÃO DO FPS PADRÃO

## Data: 27/10/2025

### 🎯 **ALTERAÇÃO SOLICITADA:**
- **FPS Padrão:** 60 → **4 FPS**

### 📁 **ARQUIVOS MODIFICADOS:**

#### **1. `app.py` - Configuração Principal:**
- **Linha 113:** `"fps_target": 4` (configuração padrão)
- **Linha 176:** `self.config["camera"]["fps_target"] = 4` (otimização GPU)
- **Linha 194:** `fps=cam_cfg.get("fps_target", 4)` (inicialização câmera)
- **Linha 289:** `self.ui.cam_fps_var.set(cam_cfg.get("fps_target", 4))` (UI)
- **Linha 858:** `self.logger.info(f"  - FPS: {cam_cfg.get('fps_target', 4)}")` (log)

#### **2. `ui_v2.py` - Interface de Usuário:**
- **Linha 185:** `self.cam_fps_var = tk.IntVar(value=4)` (valor padrão)
- **Linha 1050:** `self.cam_fps_var.set(cam_state.get("fps", 4))` (carregamento)

### ✅ **BENEFÍCIOS DO FPS = 4:**

#### **Performance:**
- **Menor uso de CPU/GPU** para processamento
- **Menor consumo de energia** da câmera
- **Processamento mais estável** das detecções

#### **Qualidade:**
- **Menos ruído** nas imagens
- **Exposição mais longa** possível
- **Detecções mais precisas** com menos movimento

#### **Estabilidade:**
- **Sistema mais robusto** para detecções
- **Menos sobrecarga** do sistema
- **Melhor para análise** de objetos estáticos

### 🔄 **SINCRONIZAÇÃO COMPLETA:**
- ✅ **Configuração padrão** (`app.py`)
- ✅ **Interface de usuário** (`ui_v2.py`)
- ✅ **Inicialização da câmera** (`app.py`)
- ✅ **Carregamento de estado** (`ui_v2.py`)
- ✅ **Logs e debug** (`app.py`)

### 📊 **FAIXA PERMITIDA:**
- **Mínimo:** 1 FPS
- **Máximo:** 200 FPS
- **Padrão:** **4 FPS** ✅

---
**Status:** ✅ **IMPLEMENTADO E SINCRONIZADO**
