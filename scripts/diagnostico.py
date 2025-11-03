#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script de diagnóstico completo para identificar problemas
"""

import sys
import os
import traceback

print("="*70)
print("DIAGNOSTICO - YOLO Detection System")
print("="*70)
print()

# Verificar Python
print("[1] Verificando Python...")
print(f"    Python: {sys.version}")
print(f"    Executavel: {sys.executable}")
print(f"    Diretorio atual: {os.getcwd()}")
print()

# Verificar imports básicos
print("[2] Testando imports básicos...")
try:
    import tkinter as tk
    print("    ✓ tkinter OK")
except Exception as e:
    print(f"    ✗ tkinter FALHOU: {e}")
    sys.exit(1)

try:
    import cv2
    print(f"    ✓ opencv OK (versao: {cv2.__version__})")
except Exception as e:
    print(f"    ✗ opencv FALHOU: {e}")

try:
    import numpy as np
    print(f"    ✓ numpy OK")
except Exception as e:
    print(f"    ✗ numpy FALHOU: {e}")

try:
    import torch
    print(f"    ✓ torch OK (versao: {torch.__version__})")
    print(f"    CUDA disponivel: {torch.cuda.is_available()}")
except Exception as e:
    print(f"    ✗ torch FALHOU: {e}")

print()

# Testar criação de janela Tkinter simples
print("[3] Testando criação de janela Tkinter...")
try:
    root = tk.Tk()
    root.title("TESTE - Janela Tkinter")
    root.geometry("400x300")
    root.deiconify()
    root.lift()
    root.focus_force()
    label = tk.Label(root, text="Se voce ve esta janela, o Tkinter funciona!", font=("Arial", 12))
    label.pack(pady=50)
    button = tk.Button(root, text="Fechar Teste", command=root.quit)
    button.pack()
    
    print("    ✓ Janela Tkinter criada")
    print("    NOTA: Uma janela de teste deve aparecer!")
    print("    Feche a janela para continuar o diagnostico...")
    root.mainloop()
    print("    ✓ Janela fechada pelo usuario")
except Exception as e:
    print(f"    ✗ Erro ao criar janela: {e}")
    traceback.print_exc()
    sys.exit(1)

print()

# Testar importação do módulo ui_v2
print("[4] Testando importacao do ui_v2...")
try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from ui_v2 import YOLODetectionUI
    print("    ✓ ui_v2 importado com sucesso")
except Exception as e:
    print(f"    ✗ Erro ao importar ui_v2: {e}")
    traceback.print_exc()
    sys.exit(1)

print()

# Testar criação da UI completa
print("[5] Testando criacao da UI completa...")
try:
    ui = YOLODetectionUI("TESTE - YOLO Detection System")
    print("    ✓ UI criada com sucesso")
    print(f"    ✓ Janela existe: {ui.root.winfo_exists()}")
    print(f"    ✓ Titulo: {ui.root.title()}")
    print(f"    ✓ Geometria: {ui.root.geometry()}")
    print()
    print("    NOTA: A janela da UI deve aparecer agora!")
    print("    Feche a janela para finalizar o diagnostico...")
    ui.run()
    print("    ✓ UI finalizada normalmente")
except Exception as e:
    print(f"    ✗ Erro ao criar/executar UI: {e}")
    traceback.print_exc()
    sys.exit(1)

print()
print("="*70)
print("DIAGNOSTICO CONCLUIDO")
print("="*70)
print()
print("Se voce chegou ate aqui sem erros, o problema pode estar em:")
print("- Inicializacao da camera")
print("- Inicializacao do detector")
print("- Execucao do app.py")
print()
input("Pressione ENTER para sair...")

