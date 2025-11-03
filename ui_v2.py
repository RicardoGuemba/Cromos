"""
Interface gráfica Tkinter para o sistema de detecção YOLO.
Versão 2.0: Layout profissional com controles à esquerda e preview maximizado.
"""

import tkinter as tk
from tkinter import ttk
import threading
import time
from typing import Optional, Callable, Dict, Any
import numpy as np
from PIL import Image, ImageTk
import cv2


class YOLODetectionUI:
    """GUI Tkinter para sistema de detecção YOLO - Versão 2.0 Profissional."""
    
    def __init__(self, window_title: str = "YOLO Detection System v2.0"):
        """Inicializa a interface."""
        self.root = tk.Tk()
        self.root.title(window_title)
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Configurar tamanho da janela otimizado
        self.root.geometry("1600x800")  # Janela maior para acomodar 3 colunas
        self.root.minsize(1400, 700)   # Mínimo adequado
        
        # Garantir que a janela apareça
        self.root.deiconify()  # Mostrar janela se estiver minimizada
        self.root.lift()      # Trazer para frente
        self.root.focus_force()  # Forçar foco
        
        # Atalhos de teclado
        self.root.bind('<Control-m>', lambda e: self.minimize_window())
        self.root.bind('<Escape>', lambda e: self.minimize_window())
        
        # Estado
        self.running = False
        self.recording = False
        self.paused = False
        
        # Debounce para parâmetros da câmera (evitar mudanças muito rápidas)
        self.camera_param_timer = None
        self.pending_camera_params = None
        
        # Callbacks (serão definidos externamente)
        self.on_start: Optional[Callable] = None
        self.on_stop: Optional[Callable] = None
        self.on_record_toggle: Optional[Callable] = None
        self.on_threshold_change: Optional[Callable] = None
        self.on_camera_param_change: Optional[Callable] = None  # NOVO
        self.on_model_toggle: Optional[Callable] = None  # NOVO
        self.on_focus_change: Optional[Callable] = None  # NOVO
        
        # Controle de som contínuo
        self.continuous_sound_active = False
        self.sound_timer = None
        
        # Frame atual
        self.current_frame: Optional[np.ndarray] = None
        self.frame_lock = threading.Lock()
        
        # Estatísticas
        self.stats = {
            "fps": 0.0,
            "smudge": 0,
            "simbolos": 0,
            "blackdot": 0,
            "transfer_count": 0,
            "avg_smudge": 0.0,
            "avg_simbolos": 0.0,
            "avg_blackdot": 0.0,
            "inference_ms": 0.0,
            "capture_fps": 0.0
        }
        
        # Sumário de estatísticas finais
        self.statistics_summary: Optional[Dict[str, Any]] = None
        
        self._build_ui()
        self._start_update_loop()
        
    def _create_tooltip(self, widget, text):
        """Cria um tooltip simples para o widget."""
        def on_enter(event):
            tooltip = tk.Toplevel()
            tooltip.wm_overrideredirect(True)
            tooltip.wm_geometry(f"+{event.x_root+10}+{event.y_root+10}")
            label = tk.Label(tooltip, text=text, background="#ffffe0", 
                           relief="solid", borderwidth=1, justify=tk.LEFT,
                           font=("Arial", 9))
            label.pack()
            widget.tooltip = tooltip
            
        def on_leave(event):
            if hasattr(widget, 'tooltip'):
                widget.tooltip.destroy()
                del widget.tooltip
                
        widget.bind('<Enter>', on_enter)
        widget.bind('<Leave>', on_leave)
    
    def _build_ui(self):
        """Constrói a interface gráfica com layout profissional."""
        # TEMA: Verde Pastel Degradê
        self.root.configure(bg="#E8F5E9")
        
        # Estilo
        style = ttk.Style()
        style.theme_use('clam')
        
        # Cores Verde Pastel
        style.configure('TFrame', background='#E8F5E9')
        style.configure('TLabelframe', background='#C8E6C9', bordercolor='#A5D6A7')
        style.configure('TLabelframe.Label', background='#C8E6C9', foreground='#1B5E20', font=('Arial', 9, 'bold'))
        style.configure('TLabel', background='#E8F5E9', foreground='#1B5E20')
        style.configure('TButton', background='#81C784', foreground='white')
        style.map('TButton', background=[('active', '#66BB6A'), ('pressed', '#4CAF50')])
        style.configure('TCheckbutton', background='#E8F5E9', foreground='#1B5E20')
        style.configure('TScale', background='#E8F5E9', troughcolor='#A5D6A7')
        
        # Configurar grid principal - 2 colunas: controles esquerda, vídeo centro
        self.root.columnconfigure(0, weight=0, minsize=220)  # Painel esquerdo
        self.root.columnconfigure(1, weight=1)  # Painel central expansível (vídeo) - MÁXIMO
        self.root.rowconfigure(0, weight=1)
        
        # === PAINEL ESQUERDO: CONTROLES PRINCIPAIS ===
        left_panel = ttk.Frame(self.root, padding="4", relief="raised", width=220)
        left_panel.grid(row=0, column=0, sticky=(tk.N, tk.S, tk.W), padx=(0, 2))
        left_panel.grid_propagate(False)
        
        # === PAINEL CENTRAL: PREVIEW ===
        center_panel = ttk.Frame(self.root, padding="5", relief="sunken", borderwidth=2)
        center_panel.grid(row=0, column=1, sticky=(tk.N, tk.S, tk.E, tk.W))
        
        # Preview maximizado - usando tk.Label como no código de referência
        self.preview_label = tk.Label(center_panel, text="Aguardando câmera...", 
                                       background="black", foreground="white",
                                       font=('Arial', 16))
        self.preview_label.pack(fill=tk.BOTH, expand=True)
        
        # === CONSTRUIR PAINEL ===
        self._build_left_panel(left_panel)
    
    
    def _build_left_panel(self, parent):
        """Constrói o painel esquerdo com controles principais."""
        
        # Título
        title_label = ttk.Label(parent, text="🎛️ Controles Principais", 
                               font=('Arial', 12, 'bold'))
        title_label.pack(pady=(0, 5))
        
        # === 1. SUMÁRIO COMPACTO ===
        stats_frame = ttk.LabelFrame(parent, text="📊 Status", padding="5")
        stats_frame.pack(fill=tk.X, pady=2)
        
        # Estatísticas em tempo real - layout compacto
        self.stats_labels = {}
        
        # Linha única com todas as informações
        self.stats_labels["transfers"] = ttk.Label(stats_frame, text="Transfers: 0", font=("Arial", 9, "bold"))
        self.stats_labels["transfers"].grid(row=0, column=0, sticky=tk.W, padx=2)
        
        self.stats_labels["roi_conf"] = ttk.Label(stats_frame, text="ROI: -", font=("Arial", 9))
        self.stats_labels["roi_conf"].grid(row=0, column=1, sticky=tk.W, padx=2)
        
        self.stats_labels["classes"] = ttk.Label(stats_frame, text="Classes: -", font=("Arial", 9))
        self.stats_labels["classes"].grid(row=0, column=2, sticky=tk.W, padx=2)
        
        # Label para classe predominante estabilizada
        self.stats_labels["predominant_class"] = ttk.Label(stats_frame, text="🎯 Predominante: -", 
                                                          font=("Arial", 9, "bold"))
        self.stats_labels["predominant_class"].grid(row=1, column=0, columnspan=3, sticky=tk.W, padx=2, pady=2)
        
        # Botão compacto para estatísticas detalhadas
        self.btn_stats = ttk.Button(stats_frame, text="📈", 
                                   command=self._open_stats_window, width=3)
        self.btn_stats.grid(row=0, column=3, sticky=tk.E, padx=2)
        
        # === 2. CONTROLES PRINCIPAIS ===
        controls_frame = ttk.LabelFrame(parent, text="⚙️ Controles do Sistema", padding="5")
        controls_frame.pack(fill=tk.X, pady=2)
        
        # Primeira linha: Controles principais de captura
        controls_row1 = ttk.Frame(controls_frame)
        controls_row1.pack(fill=tk.X, pady=(0, 2))
        
        self.btn_start = ttk.Button(controls_row1, text="▶ Iniciar Captura", 
                                   command=self._on_start_click, width=20)
        self.btn_start.pack(side=tk.LEFT, padx=2)
        self._create_tooltip(self.btn_start, 
                           "INICIAR CAPTURA:\n"
                           "• Ativa a câmera e inicia o processamento\n"
                           "• Os modelos YOLO começam a detectar objetos\n"
                           "• O botão de gravação será habilitado após iniciar")
        
        self.btn_stop = ttk.Button(controls_row1, text="⏹ Parar Captura", 
                                   command=self._on_stop_click, state=tk.DISABLED, width=20)
        self.btn_stop.pack(side=tk.LEFT, padx=2)
        self._create_tooltip(self.btn_stop, 
                           "PARAR CAPTURA:\n"
                           "• Desativa a câmera e para todo o processamento\n"
                           "• Todos os modelos são desativados\n"
                           "• A gravação também será interrompida se estiver ativa")
        
        # Segunda linha: Gravação e Pausa
        controls_row2 = ttk.Frame(controls_frame)
        controls_row2.pack(fill=tk.X, pady=2)
        
        # Botão de Gravação - Destaque visual maior com explicação clara
        self.btn_record = ttk.Button(controls_row2, text="🔴 Iniciar Gravação de Vídeo", 
                                     command=self._on_record_click, state=tk.DISABLED, width=28)
        self.btn_record.pack(side=tk.LEFT, padx=2)
        # Tooltip explicativo detalhado
        self._create_tooltip(self.btn_record, 
                           "GRAVAÇÃO DE VÍDEO:\n"
                           "• Primeiro clique: INICIA a gravação\n"
                           "• Segundo clique: PARA a gravação\n"
                           "• O vídeo é salvo automaticamente na pasta 'recordings'\n"
                           "• O vídeo terá a mesma velocidade da câmera (FPS natural)\n"
                           "• Funciona apenas quando a captura está ativa")
        
        self.btn_pause = ttk.Button(controls_row2, text="⏸ Pausar Processamento", 
                                    command=self._on_pause_click, state=tk.DISABLED, width=22)
        self.btn_pause.pack(side=tk.LEFT, padx=2)
        self._create_tooltip(self.btn_pause, 
                           "PAUSAR PROCESSAMENTO:\n"
                           "• Pausa temporariamente as detecções YOLO\n"
                           "• A câmera continua capturando frames\n"
                           "• Clique novamente para RETOMAR o processamento\n"
                           "• Útil para inspecionar frames sem processamento")
        
        # Botão minimizar compacto
        btn_minimize = ttk.Button(controls_row2, text="🗕", 
                                 command=self.minimize_window, width=5)
        btn_minimize.pack(side=tk.LEFT, padx=2)
        self._create_tooltip(btn_minimize, "Minimizar janela (Ctrl+M ou ESC)")
        
        # === 3. PARÂMETROS DA CÂMERA BASLER ===
        camera_frame = ttk.LabelFrame(parent, text="📷 Câmera Basler - Controles Avançados", padding="10")
        camera_frame.pack(fill=tk.X, pady=(0, 8))
        
        # Status de aplicação (row 0)
        self.camera_apply_status = ttk.Label(camera_frame, text="", font=('Arial', 9), foreground="orange")
        self.camera_apply_status.grid(row=0, column=0, columnspan=4, sticky=tk.W, pady=(0, 8))
        
        # FPS Target (row 1) - Layout melhorado
        ttk.Label(camera_frame, text="FPS:", font=("Arial", 9, "bold")).grid(row=1, column=0, sticky=tk.W, pady=(0, 4))
        self.cam_fps_var = tk.IntVar(value=4)  # Padrão 4 FPS
        self.cam_fps_scale = ttk.Scale(camera_frame, from_=1, to=200, 
                                       variable=self.cam_fps_var, orient=tk.HORIZONTAL,
                                       command=self._on_camera_param_change, length=40)
        self.cam_fps_scale.grid(row=1, column=1, columnspan=2, sticky=(tk.W, tk.E), padx=(8, 4), pady=(0, 4))
        self.cam_fps_label = ttk.Label(camera_frame, text="30", width=6, font=("Arial", 9, "bold"))
        self.cam_fps_label.grid(row=1, column=3, sticky=tk.W, pady=(0, 4))
        
        # Exposição (row 2) - Layout melhorado
        ttk.Label(camera_frame, text="Exposição (µs):", font=("Arial", 9, "bold")).grid(row=2, column=0, sticky=tk.W, pady=(0, 4))
        self.cam_exposure_var = tk.IntVar(value=10000)  # Padrão 10ms (10000µs)
        self.cam_exposure_scale = ttk.Scale(camera_frame, from_=10, to=100000, 
                                            variable=self.cam_exposure_var, orient=tk.HORIZONTAL,
                                            command=self._on_camera_param_change, length=40)
        self.cam_exposure_scale.grid(row=2, column=1, columnspan=2, sticky=(tk.W, tk.E), padx=(8, 4), pady=(0, 4))
        self.cam_exposure_label = ttk.Label(camera_frame, text="10000", width=6, font=("Arial", 9, "bold"))
        self.cam_exposure_label.grid(row=2, column=3, sticky=tk.W, pady=(0, 4))
        
        # Ganho (row 3) - Layout melhorado
        ttk.Label(camera_frame, text="Ganho (dB):", font=("Arial", 9, "bold")).grid(row=3, column=0, sticky=tk.W, pady=(0, 4))
        self.cam_gain_var = tk.DoubleVar(value=0.0)
        self.cam_gain_scale = ttk.Scale(camera_frame, from_=0, to=48, 
                                        variable=self.cam_gain_var, orient=tk.HORIZONTAL,
                                        command=self._on_camera_param_change, length=40)
        self.cam_gain_scale.grid(row=3, column=1, columnspan=2, sticky=(tk.W, tk.E), padx=(8, 4), pady=(0, 4))
        self.cam_gain_label = ttk.Label(camera_frame, text="0.0", width=6, font=("Arial", 9, "bold"))
        self.cam_gain_label.grid(row=3, column=3, sticky=tk.W, pady=(0, 4))
        
        # Balance White Auto (row 4) - Layout melhorado
        ttk.Label(camera_frame, text="Balance:", font=("Arial", 9, "bold")).grid(row=4, column=0, sticky=tk.W, pady=(0, 4))
        self.cam_balance_var = tk.StringVar(value="Off")
        self.cam_balance_combo = ttk.Combobox(camera_frame, textvariable=self.cam_balance_var,
                                              values=["Off", "Once", "Continuous"],
                                              state="readonly", width=12, font=("Arial", 9))
        self.cam_balance_combo.grid(row=4, column=1, columnspan=2, sticky=(tk.W, tk.E), padx=(8, 4), pady=(0, 4))
        self.cam_balance_combo.bind("<<ComboboxSelected>>", self._on_balance_change)
        
        # Resolução (row 5) - Layout melhorado
        ttk.Label(camera_frame, text="Resolução:", font=("Arial", 9, "bold")).grid(row=5, column=0, sticky=tk.W, pady=(0, 4))
        self.cam_resolution_var = tk.StringVar(value="1280x720 (HD)")
        self.cam_resolution_combo = ttk.Combobox(camera_frame, textvariable=self.cam_resolution_var,
                                                 values=["640x480 (VGA)", "800x600 (SVGA)", "1024x768 (XGA)", 
                                                        "1280x720 (HD)", "1920x1080 (Full HD)", "1440x1080"],
                                                 state="readonly", width=15, font=("Arial", 9))
        self.cam_resolution_combo.grid(row=5, column=1, columnspan=2, sticky=(tk.W, tk.E), padx=(8, 4), pady=(0, 4))
        self.cam_resolution_combo.bind("<<ComboboxSelected>>", self._on_resolution_change)
        
        # === CONTROLES DE AJUSTE AUTOMÁTICO ===
        # Separador interno
        ttk.Separator(camera_frame, orient='horizontal').grid(row=6, column=0, columnspan=4, sticky=(tk.W, tk.E), pady=(8, 6))
        
        # Título da seção de ajuste automático
        auto_title = ttk.Label(camera_frame, text="🎛️ Ajuste Automático", font=("Arial", 9, "bold"))
        auto_title.grid(row=7, column=0, columnspan=4, sticky=tk.W, pady=(0, 6))
        
        # Auto Exposição (row 8)
        self.auto_exposure_var = tk.BooleanVar(value=False)
        self.auto_exposure_check = ttk.Checkbutton(camera_frame, text="Auto Exposição", variable=self.auto_exposure_var,
                                                  command=self._on_auto_exposure_change)
        self.auto_exposure_check.grid(row=8, column=0, columnspan=2, sticky=tk.W, pady=(0, 3))
        
        # Auto Ganho (row 9)
        self.auto_gain_var = tk.BooleanVar(value=False)
        self.auto_gain_check = ttk.Checkbutton(camera_frame, text="Auto Ganho", variable=self.auto_gain_var,
                                              command=self._on_auto_gain_change)
        self.auto_gain_check.grid(row=9, column=0, columnspan=2, sticky=tk.W, pady=(0, 3))
        
        # Status do ajuste automático
        self.auto_status_label = ttk.Label(camera_frame, text="", font=('Arial', 8), foreground="blue")
        self.auto_status_label.grid(row=10, column=0, columnspan=4, sticky=tk.W, pady=(4, 0))
        
        # Configurar colunas do frame da câmera
        camera_frame.columnconfigure(1, weight=1)
        camera_frame.columnconfigure(2, weight=1)
        
        # Separador visual
        separator = ttk.Separator(parent, orient='horizontal')
        separator.pack(fill=tk.X, pady=(8, 5))
        
        # === 4. THRESHOLDS DE DETECÇÃO ===
        detection_frame = ttk.LabelFrame(parent, text="🎯 Thresholds de Detecção", padding="10")
        detection_frame.pack(fill=tk.X, pady=(0, 8))
        
        # ROI - Layout melhorado
        ttk.Label(detection_frame, text="ROI:", font=("Arial", 9, "bold")).grid(row=0, column=0, sticky=tk.W, pady=(0, 4))
        self.roi_conf_var = tk.DoubleVar(value=50)  # 50% = 0.5
        self.roi_conf_slider = ttk.Scale(detection_frame, from_=0, to=100,
                                         variable=self.roi_conf_var, orient=tk.HORIZONTAL,
                                         command=self._on_threshold_change, length=40)
        self.roi_conf_slider.grid(row=0, column=1, columnspan=2, sticky=(tk.W, tk.E), padx=(8, 4), pady=(0, 4))
        self.roi_conf_label = ttk.Label(detection_frame, text="0.50", width=6, font=("Arial", 9, "bold"))
        self.roi_conf_label.grid(row=0, column=3, sticky=tk.W, pady=(0, 4))
        
        # Smudge - Layout melhorado
        ttk.Label(detection_frame, text="Smudge:", font=("Arial", 9, "bold")).grid(row=1, column=0, sticky=tk.W, pady=(0, 4))
        self.smudge_conf_var = tk.DoubleVar(value=50)  # 50% = 0.5
        self.smudge_conf_slider = ttk.Scale(detection_frame, from_=0, to=100,
                                            variable=self.smudge_conf_var, orient=tk.HORIZONTAL,
                                            command=self._on_threshold_change, length=40)
        self.smudge_conf_slider.grid(row=1, column=1, columnspan=2, sticky=(tk.W, tk.E), padx=(8, 4), pady=(0, 4))
        self.smudge_conf_label = ttk.Label(detection_frame, text="0.50", width=6, font=("Arial", 9, "bold"))
        self.smudge_conf_label.grid(row=1, column=3, sticky=tk.W, pady=(0, 4))
        
        # Símbolos - Layout melhorado
        ttk.Label(detection_frame, text="Símbolos:", font=("Arial", 9, "bold")).grid(row=2, column=0, sticky=tk.W, pady=(0, 4))
        self.simbolos_conf_var = tk.DoubleVar(value=50)  # 50% = 0.5
        self.simbolos_conf_slider = ttk.Scale(detection_frame, from_=0, to=100,
                                              variable=self.simbolos_conf_var, orient=tk.HORIZONTAL,
                                              command=self._on_threshold_change, length=40)
        self.simbolos_conf_slider.grid(row=2, column=1, columnspan=2, sticky=(tk.W, tk.E), padx=(8, 4), pady=(0, 4))
        self.simbolos_conf_label = ttk.Label(detection_frame, text="0.50", width=6, font=("Arial", 9, "bold"))
        self.simbolos_conf_label.grid(row=2, column=3, sticky=tk.W, pady=(0, 4))
        
        # BlackDot - Layout melhorado
        ttk.Label(detection_frame, text="BlackDot:", font=("Arial", 9, "bold")).grid(row=3, column=0, sticky=tk.W, pady=(0, 4))
        self.blackdot_conf_var = tk.DoubleVar(value=50)  # 50% = 0.5
        self.blackdot_conf_slider = ttk.Scale(detection_frame, from_=0, to=100,
                                              variable=self.blackdot_conf_var, orient=tk.HORIZONTAL,
                                              command=self._on_threshold_change, length=40)
        self.blackdot_conf_slider.grid(row=3, column=1, columnspan=2, sticky=(tk.W, tk.E), padx=(8, 4), pady=(0, 4))
        self.blackdot_conf_label = ttk.Label(detection_frame, text="0.50", width=6, font=("Arial", 9, "bold"))
        self.blackdot_conf_label.grid(row=3, column=3, sticky=tk.W, pady=(0, 4))
        
        # Configurar colunas do frame de detecção
        detection_frame.columnconfigure(1, weight=1)
        detection_frame.columnconfigure(2, weight=1)
        
        # === 5. CONTROLES DE FOCO E NITIDEZ === (REMOVIDO - AJUSTE AUTOMÁTICO)
        # Seção removida conforme solicitado - sistema usa ajuste automático
        
        # === 6. CONTROLES DE MODELOS ===
        models_frame = ttk.LabelFrame(parent, text="🤖 Controles de Modelos", padding="10")
        models_frame.pack(fill=tk.X, pady=(0, 8))
        
        # Checkboxes para ativar/desativar modelos - Layout melhorado
        self.model_vars = {}
        
        # ROI (Segmentação)
        self.model_vars["seg"] = tk.BooleanVar(value=True)
        seg_check = ttk.Checkbutton(models_frame, text="ROI", 
                                   variable=self.model_vars["seg"],
                                   command=lambda: self._on_model_toggle("seg", self.model_vars["seg"].get()))
        seg_check.grid(row=0, column=0, sticky=tk.W, pady=(0, 4))
        
        # Smudge
        self.model_vars["smudge"] = tk.BooleanVar(value=True)
        smudge_check = ttk.Checkbutton(models_frame, text="Smudge", 
                                      variable=self.model_vars["smudge"],
                                      command=lambda: self._on_model_toggle("smudge", self.model_vars["smudge"].get()))
        smudge_check.grid(row=0, column=1, sticky=tk.W, pady=(0, 4))
        
        # Símbolos
        self.model_vars["simbolos"] = tk.BooleanVar(value=True)
        simbolos_check = ttk.Checkbutton(models_frame, text="Símbolos", 
                                        variable=self.model_vars["simbolos"],
                                        command=lambda: self._on_model_toggle("simbolos", self.model_vars["simbolos"].get()))
        simbolos_check.grid(row=1, column=0, sticky=tk.W, pady=(0, 4))
        
        # BlackDot
        self.model_vars["blackdot"] = tk.BooleanVar(value=True)
        blackdot_check = ttk.Checkbutton(models_frame, text="BlackDot", 
                                        variable=self.model_vars["blackdot"],
                                        command=lambda: self._on_model_toggle("blackdot", self.model_vars["blackdot"].get()))
        blackdot_check.grid(row=1, column=1, sticky=tk.W, pady=(0, 4))
        
        # Botões de controle rápido - Layout melhorado
        btn_frame = ttk.Frame(models_frame)
        btn_frame.grid(row=2, column=0, columnspan=2, pady=(8, 0))
        
        btn_all_on = ttk.Button(btn_frame, text="✅ Todos ON", 
                               command=self._enable_all_models, width=12)
        btn_all_on.pack(side=tk.LEFT, padx=(0, 4))
        
        btn_all_off = ttk.Button(btn_frame, text="❌ Todos OFF", 
                                command=self._disable_all_models, width=12)
        btn_all_off.pack(side=tk.LEFT, padx=(0, 4))
        
        # Configurar colunas do frame de modelos
        models_frame.columnconfigure(0, weight=1)
        models_frame.columnconfigure(1, weight=1)
        
        # === 7. ESTATÍSTICAS ===
        stats_frame = ttk.LabelFrame(parent, text="📊 Stats", padding="6")
        stats_frame.pack(fill=tk.X, pady=(0, 4))
        
        # Performance - Layout ultra compacto
        perf_grid = ttk.Frame(stats_frame)
        perf_grid.pack(fill=tk.X, pady=(0, 4))
        
        ttk.Label(perf_grid, text="FPS:", font=('Arial', 8, 'bold')).grid(row=0, column=0, sticky=tk.W, pady=1)
        self.fps_label = ttk.Label(perf_grid, text="0.0", font=('Arial', 8, 'bold'), foreground="blue")
        self.fps_label.grid(row=0, column=1, sticky=tk.W, padx=(2, 6))
        
        ttk.Label(perf_grid, text="Cap:", font=('Arial', 8, 'bold')).grid(row=0, column=2, sticky=tk.W, pady=1)
        self.capture_fps_label = ttk.Label(perf_grid, text="0.0", font=('Arial', 8, 'bold'), foreground="#00AA00")
        self.capture_fps_label.grid(row=0, column=3, sticky=tk.W, padx=(2, 6))
        
        ttk.Label(perf_grid, text="Lat:", font=('Arial', 8, 'bold')).grid(row=0, column=4, sticky=tk.W, pady=1)
        self.infer_label = ttk.Label(perf_grid, text="0ms", font=('Arial', 8, 'bold'), foreground="purple")
        self.infer_label.grid(row=0, column=5, sticky=tk.W, padx=(2, 0))
        
        # Detecções - Layout ultra compacto
        det_grid = ttk.Frame(stats_frame)
        det_grid.pack(fill=tk.X, pady=(0, 4))
        
        ttk.Label(det_grid, text="Smudge:", font=('Arial', 8, 'bold')).grid(row=0, column=0, sticky=tk.W, pady=1)
        self.smudge_label = ttk.Label(det_grid, text="0", foreground="red", font=('Arial', 8, 'bold'))
        self.smudge_label.grid(row=0, column=1, sticky=tk.W, padx=(2, 6))
        
        ttk.Label(det_grid, text="Símbolos:", font=('Arial', 8, 'bold')).grid(row=0, column=2, sticky=tk.W, pady=1)
        self.simbolos_label = ttk.Label(det_grid, text="0", foreground="blue", font=('Arial', 8, 'bold'))
        self.simbolos_label.grid(row=0, column=3, sticky=tk.W, padx=(2, 6))
        
        ttk.Label(det_grid, text="BlackDot:", font=('Arial', 8, 'bold')).grid(row=0, column=4, sticky=tk.W, pady=1)
        self.blackdot_label = ttk.Label(det_grid, text="0", foreground="orange", font=('Arial', 8, 'bold'))
        self.blackdot_label.grid(row=0, column=5, sticky=tk.W, padx=(2, 0))
        
        # Transfer - Layout ultra compacto
        transfer_grid = ttk.Frame(stats_frame)
        transfer_grid.pack(fill=tk.X, pady=(0, 4))
        
        ttk.Label(transfer_grid, text="Transfer:", font=('Arial', 8, 'bold')).grid(row=0, column=0, sticky=tk.W, pady=1)
        self.transfer_label = ttk.Label(transfer_grid, text="0", font=('Arial', 8, 'bold'), foreground="#00AA00")
        self.transfer_label.grid(row=0, column=1, sticky=tk.W, padx=(2, 0))
        
        # Estatísticas de aprovação
        ttk.Label(transfer_grid, text="Avaliados:", font=('Arial', 8, 'bold')).grid(row=1, column=0, sticky=tk.W, pady=1)
        self.evaluated_label = ttk.Label(transfer_grid, text="0", font=('Arial', 8, 'bold'), foreground="#2196F3")
        self.evaluated_label.grid(row=1, column=1, sticky=tk.W, padx=(2, 0))
        
        ttk.Label(transfer_grid, text="Aprovados:", font=('Arial', 8, 'bold')).grid(row=2, column=0, sticky=tk.W, pady=1)
        self.approved_label = ttk.Label(transfer_grid, text="0", font=('Arial', 8, 'bold'), foreground="#4CAF50")
        self.approved_label.grid(row=2, column=1, sticky=tk.W, padx=(2, 0))
        
        ttk.Label(transfer_grid, text="Reprovados:", font=('Arial', 8, 'bold')).grid(row=3, column=0, sticky=tk.W, pady=1)
        self.rejected_label = ttk.Label(transfer_grid, text="0", font=('Arial', 8, 'bold'), foreground="#F44336")
        self.rejected_label.grid(row=3, column=1, sticky=tk.W, padx=(2, 0))
        
        ttk.Label(transfer_grid, text="Taxa Aprovação:", font=('Arial', 8, 'bold')).grid(row=4, column=0, sticky=tk.W, pady=1)
        self.approval_rate_label = ttk.Label(transfer_grid, text="0.0%", font=('Arial', 8, 'bold'), foreground="#FF9800")
        self.approval_rate_label.grid(row=4, column=1, sticky=tk.W, padx=(2, 0))
        
        # Estatísticas de classes detectadas médias
        ttk.Label(transfer_grid, text="Smudge Médio:", font=('Arial', 8, 'bold')).grid(row=5, column=0, sticky=tk.W, pady=1)
        self.avg_smudge_detected_label = ttk.Label(transfer_grid, text="0.0%", font=('Arial', 8, 'bold'), foreground="#9C27B0")
        self.avg_smudge_detected_label.grid(row=5, column=1, sticky=tk.W, padx=(2, 0))
        
        ttk.Label(transfer_grid, text="Símbolo Médio:", font=('Arial', 8, 'bold')).grid(row=6, column=0, sticky=tk.W, pady=1)
        self.avg_simbolos_detected_label = ttk.Label(transfer_grid, text="0.0%", font=('Arial', 8, 'bold'), foreground="#3F51B5")
        self.avg_simbolos_detected_label.grid(row=6, column=1, sticky=tk.W, padx=(2, 0))
        
        ttk.Label(transfer_grid, text="BlackDot Médio:", font=('Arial', 8, 'bold')).grid(row=7, column=0, sticky=tk.W, pady=1)
        self.avg_blackdot_detected_label = ttk.Label(transfer_grid, text="0.0%", font=('Arial', 8, 'bold'), foreground="#795548")
        self.avg_blackdot_detected_label.grid(row=7, column=1, sticky=tk.W, padx=(2, 0))
        
        # === 8. CONTROLES DE CLASSES E IOU ===
        # === 4. CONTROLES DE CLASSE ===
        classes_frame = ttk.LabelFrame(parent, text="🏷️ Controle de Classes", padding="6")
        classes_frame.pack(fill=tk.X, pady=(0, 4))
        
        # ROI Classes
        roi_classes_frame = ttk.LabelFrame(classes_frame, text="ROI", padding="4")
        roi_classes_frame.pack(fill=tk.X, pady=(0, 4))
        
        self.roi_class_vars = {}
        roi_classes = ["Smudge", "Fluminense", "Palmeiras"]
        for i, class_name in enumerate(roi_classes):
            self.roi_class_vars[class_name] = tk.BooleanVar(value=True)
            ttk.Checkbutton(roi_classes_frame, text=class_name, 
                           variable=self.roi_class_vars[class_name],
                           command=lambda c=class_name: self._on_class_toggle("roi", c, self.roi_class_vars[c].get())).grid(row=0, column=i, sticky=tk.W, padx=(0, 8))
        
        # Símbolos Classes
        simbolos_classes_frame = ttk.LabelFrame(classes_frame, text="Símbolos", padding="4")
        simbolos_classes_frame.pack(fill=tk.X, pady=(0, 4))
        
        self.simbolos_class_vars = {}
        # Classes do modelo best.pt: FIFA, Simbolo, String (cada com OK e NO) - 6 classes
        simbolos_classes = ["FIFA_NO", "FIFA_OK", "Simbolo_NO", "Simbolo_OK", "String_NO", "String_OK"]
        for i, class_name in enumerate(simbolos_classes):
            self.simbolos_class_vars[class_name] = tk.BooleanVar(value=True)
            ttk.Checkbutton(simbolos_classes_frame, text=class_name, 
                           variable=self.simbolos_class_vars[class_name],
                           command=lambda c=class_name: self._on_class_toggle("simbolos", c, self.simbolos_class_vars[c].get())).grid(row=0, column=i, sticky=tk.W, padx=(0, 4))
        
        # Botões de controle rápido para classes
        class_btn_frame = ttk.Frame(classes_frame)
        class_btn_frame.pack(fill=tk.X, pady=(4, 0))
        
        ttk.Button(class_btn_frame, text="✅ ON", command=self._enable_all_classes, width=6).pack(side=tk.LEFT, padx=(0, 2))
        ttk.Button(class_btn_frame, text="❌ OFF", command=self._disable_all_classes, width=6).pack(side=tk.LEFT, padx=(0, 2))
        ttk.Button(class_btn_frame, text="🔄", command=self._reset_classes, width=4).pack(side=tk.LEFT)
        
        # Médias - Layout compacto
        avg_frame = ttk.LabelFrame(stats_frame, text="📈 Médias", padding="4")
        avg_frame.pack(fill=tk.X, pady=(0, 4))
        
        avg_grid = ttk.Frame(avg_frame)
        avg_grid.pack(fill=tk.X)
        
        ttk.Label(avg_grid, text="Smudge:", font=('Arial', 8, 'bold')).grid(row=0, column=0, sticky=tk.W, pady=1)
        self.avg_smudge_label = ttk.Label(avg_grid, text="0.0", font=('Arial', 8, 'bold'), foreground="red")
        self.avg_smudge_label.grid(row=0, column=1, sticky=tk.W, padx=(4, 8))
        
        ttk.Label(avg_grid, text="Símbolos:", font=('Arial', 8, 'bold')).grid(row=0, column=2, sticky=tk.W, pady=1)
        self.avg_simbolos_label = ttk.Label(avg_grid, text="0.0", font=('Arial', 8, 'bold'), foreground="blue")
        self.avg_simbolos_label.grid(row=0, column=3, sticky=tk.W, padx=(4, 8))
        
        ttk.Label(avg_grid, text="BlackDot:", font=('Arial', 8, 'bold')).grid(row=0, column=4, sticky=tk.W, pady=1)
        self.avg_blackdot_label = ttk.Label(avg_grid, text="0.0", font=('Arial', 8, 'bold'), foreground="orange")
        self.avg_blackdot_label.grid(row=0, column=5, sticky=tk.W, padx=(4, 0))
        
        # Status - Layout compacto
        status_frame = ttk.Frame(stats_frame)
        status_frame.pack(fill=tk.X, pady=(4, 0))
        
        self.status_label = ttk.Label(status_frame, text="● Pronto", font=('Arial', 9, 'bold'), foreground="gray")
        self.status_label.pack(side=tk.LEFT)
        
        # Info sobre resolução
        info_label = ttk.Label(status_frame, 
                              text="ℹ️ 640px | 1280x720 HD",
                              font=('Arial', 7), foreground="blue")
        info_label.pack(side=tk.RIGHT)
    
    def _on_start_click(self):
        """Handler do botão Iniciar."""
        if self.on_start:
            self.on_start()
        self.running = True
        self.btn_start.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        self.btn_record.config(state=tk.NORMAL)
        self.btn_pause.config(state=tk.NORMAL)
        self.set_status("● Executando", "green")
    
    def _on_stop_click(self):
        """Handler do botão Parar."""
        if self.on_stop:
            self.on_stop()
        self.running = False
        self.recording = False
        self.btn_start.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)
        self.btn_record.config(state=tk.DISABLED)
        self.btn_pause.config(state=tk.DISABLED)
        self.btn_record.config(text="🔴 Iniciar Gravação")
        self.set_status("● Parado", "red")
    
    def _on_record_click(self):
        """Handler do botão Gravar."""
        self.recording = not self.recording
        if self.on_record_toggle:
            self.on_record_toggle(self.recording)
        
        if self.recording:
            self.btn_record.config(text="⏹ Parar Gravação")
            self.set_status("● Gravando vídeo...", "red")
        else:
            self.btn_record.config(text="🔴 Iniciar Gravação")
            self.set_status("● Executando", "green")
    
    def _on_pause_click(self):
        """Handler do botão Pausar."""
        self.paused = not self.paused
        if self.paused:
            self.btn_pause.config(text="▶ Retomar")
            self.set_status("● Pausado", "orange")
        else:
            self.btn_pause.config(text="⏸ Pausar")
            self.set_status("● Executando", "green")
    
    def _on_threshold_change(self, _=None):
        """Handler de mudança nos thresholds de detecção."""
        # Converter valores de 0-100 para 0.0-1.0
        roi_val = self.roi_conf_var.get() / 100.0
        smudge_val = self.smudge_conf_var.get() / 100.0
        simbolos_val = self.simbolos_conf_var.get() / 100.0
        blackdot_val = self.blackdot_conf_var.get() / 100.0
        
        # Atualizar labels com valores convertidos
        self.roi_conf_label.config(text=f"{roi_val:.2f}")
        self.smudge_conf_label.config(text=f"{smudge_val:.2f}")
        self.simbolos_conf_label.config(text=f"{simbolos_val:.2f}")
        self.blackdot_conf_label.config(text=f"{blackdot_val:.2f}")
        
        # Callback
        if self.on_threshold_change:
            thresholds = {
                "roi_conf": roi_val,
                "smudge_conf": smudge_val,
                "simbolo_conf": simbolos_val,
                "blackdot_conf": blackdot_val
            }
            self.on_threshold_change(thresholds)
    
    def _on_camera_param_change(self, _=None):
        """Handler de mudança nos parâmetros da câmera com debounce."""
        # Atualizar labels imediatamente para feedback visual
        self.cam_fps_label.config(text=str(int(self.cam_fps_var.get())))
        self.cam_exposure_label.config(text=str(int(self.cam_exposure_var.get())))
        self.cam_gain_label.config(text=f"{self.cam_gain_var.get():.1f}")
        
        # Mostrar status de aguardando
        self.camera_apply_status.config(text="⏳ Aguardando aplicação...")
        
        # Guardar parâmetros pendentes
        self.pending_camera_params = {
            "fps": int(self.cam_fps_var.get()),
            "exposure": int(self.cam_exposure_var.get()),
            "gain": self.cam_gain_var.get()
        }
        
        # Cancelar timer anterior se existir
        if self.camera_param_timer is not None:
            self.root.after_cancel(self.camera_param_timer)
        
        # Agendar aplicação após 500ms (debounce)
        self.camera_param_timer = self.root.after(500, self._apply_camera_params)
    
    def _on_balance_change(self, _=None):
        """Handler de mudança no Balance White Auto."""
        balance_mode = self.cam_balance_var.get()
        self.camera_apply_status.config(text=f"✓ Balance: {balance_mode}", foreground="green")
        
        if self.on_camera_param_change:
            self.on_camera_param_change({"balance_white_auto": balance_mode})
    
    def _on_resolution_change(self, _=None):
        """Handler de mudança na resolução."""
        resolution_str = self.cam_resolution_var.get()
        # Extrair largura x altura da string "1280x720 (HD)"
        resolution = resolution_str.split(' ')[0]  # Pega "1280x720"
        self.camera_apply_status.config(text=f"⚠ Resolução: {resolution} - Reinicie a câmera!", foreground="orange")
        
        # Resolução não é aplicada diretamente, apenas informativa
        # Limpar status após 3s
        self.root.after(3000, lambda: self.camera_apply_status.config(text="", foreground="orange"))
    
    def _on_auto_exposure_change(self):
        """Handler de mudança no ajuste automático de exposição."""
        auto_enabled = self.auto_exposure_var.get()
        if auto_enabled:
            self.auto_status_label.config(text="🔄 Auto Exposição: ATIVADO - Ajustando automaticamente", foreground="green")
            # Desabilitar slider manual
            self.cam_exposure_scale.config(state="disabled")
        else:
            self.auto_status_label.config(text="⚙️ Auto Exposição: DESATIVADO - Controle manual", foreground="blue")
            # Habilitar slider manual
            self.cam_exposure_scale.config(state="normal")
        
        # Chamar callback se disponível
        if hasattr(self, 'on_auto_camera_change'):
            self.on_auto_camera_change({
                'auto_exposure': auto_enabled,
                'auto_gain': self.auto_gain_var.get()
            })
    
    def _on_auto_gain_change(self):
        """Handler de mudança no ajuste automático de ganho."""
        auto_enabled = self.auto_gain_var.get()
        if auto_enabled:
            self.auto_status_label.config(text="🔄 Auto Ganho: ATIVADO - Ajustando automaticamente", foreground="green")
            # Desabilitar slider manual
            self.cam_gain_scale.config(state="disabled")
        else:
            self.auto_status_label.config(text="⚙️ Auto Ganho: DESATIVADO - Controle manual", foreground="blue")
            # Habilitar slider manual
            self.cam_gain_scale.config(state="normal")
        
        # Chamar callback se disponível
        if hasattr(self, 'on_auto_camera_change'):
            self.on_auto_camera_change({
                'auto_exposure': self.auto_exposure_var.get(),
                'auto_gain': auto_enabled
            })
    
    def _on_class_toggle(self, model_type: str, class_name: str, enabled: bool):
        """Handler de mudança em classes específicas."""
        status = "ATIVADA" if enabled else "DESATIVADA"
        print(f"Classe {class_name} ({model_type}): {status}")
        
        # Chamar callback se disponível
        if hasattr(self, 'on_class_change'):
            self.on_class_change({
                'model_type': model_type,
                'class_name': class_name,
                'enabled': enabled
            })
    
    def _enable_all_classes(self):
        """Ativa todas as classes."""
        for var in self.roi_class_vars.values():
            var.set(True)
        for var in self.simbolos_class_vars.values():
            var.set(True)
        print("✅ Todas as classes ativadas")
    
    def _disable_all_classes(self):
        """Desativa todas as classes."""
        for var in self.roi_class_vars.values():
            var.set(False)
        for var in self.simbolos_class_vars.values():
            var.set(False)
        print("❌ Todas as classes desativadas")
    
    def _reset_classes(self):
        """Reset para configuração padrão (todas ativas)."""
        self._enable_all_classes()
        print("🔄 Classes resetadas para padrão")
    
    def _on_iou_change(self, _=None):
        """Handler de mudança nos valores de IOU."""
        smudge_iou = self.smudge_iou_var.get()
        simbolos_iou = self.simbolos_iou_var.get()
        blackdot_iou = self.blackdot_iou_var.get()
        
        # Atualizar labels
        self.smudge_iou_label.config(text=f"{smudge_iou:.2f}")
        self.simbolos_iou_label.config(text=f"{simbolos_iou:.2f}")
        self.blackdot_iou_label.config(text=f"{blackdot_iou:.2f}")
        
        # Callback para atualizar IOU no detector
        if self.on_threshold_change:
            self.on_threshold_change({
                "smudge_iou": smudge_iou,
                "simbolos_iou": simbolos_iou,
                "blackdot_iou": blackdot_iou
            })
    
    def _on_overlay_change(self):
        """Handler de mudança nos controles de overlay."""
        smudge_overlay = self.smudge_overlay_var.get()
        simbolos_overlay = self.simbolos_overlay_var.get()
        blackdot_overlay = self.blackdot_overlay_var.get()
        
        # Callback para atualizar overlay no detector
        if self.on_threshold_change:
            self.on_threshold_change({
                "smudge_overlay": smudge_overlay,
                "simbolos_overlay": simbolos_overlay,
                "blackdot_overlay": blackdot_overlay
            })
    
    def _apply_camera_params(self):
        """Aplica os parâmetros da câmera (chamado após debounce)."""
        if self.on_camera_param_change and self.pending_camera_params:
            self.camera_apply_status.config(text="✓ Aplicado!", foreground="green")
            self.on_camera_param_change(self.pending_camera_params)
            self.pending_camera_params = None
            # Limpar status após 2s
            self.root.after(2000, lambda: self.camera_apply_status.config(text="", foreground="orange"))
        self.camera_param_timer = None
    
    def update_frame(self, frame: np.ndarray):
        """Atualiza o frame exibido no preview."""
        with self.frame_lock:
            self.current_frame = frame.copy()
    
    def update_stats(self, stats: Dict[str, Any]):
        """Atualiza as estatísticas exibidas."""
        self.stats.update(stats)
    
    def update_statistics_summary(self, summary: Dict[str, Any]):
        """
        Atualiza o sumário final de estatísticas e exibe na janela de estatísticas detalhadas.
        
        Args:
            summary: Dicionário com sumário de estatísticas retornado por get_final_statistics_summary()
        """
        self.statistics_summary = summary
        self._update_summary_display()
    
    def _update_summary_display(self):
        """Atualiza a exibição do sumário na janela de estatísticas detalhadas."""
        if not hasattr(self, 'summary_text') or not self.statistics_summary:
            return
        
        summary = self.statistics_summary
        text_content = []
        
        # Cabeçalho
        text_content.append("=" * 70)
        text_content.append("📊 SUMÁRIO FINAL DE ESTATÍSTICAS")
        text_content.append("=" * 70)
        text_content.append("")
        
        # Total de transfers avaliados
        total_transfers = summary.get("total_transfers_evaluated", 0)
        text_content.append(f"🔄 Transfers Avaliados: {total_transfers}")
        text_content.append("")
        
        if total_transfers > 0:
            # Estatísticas por classe
            text_content.append("📦 Objetos Detectados por Classe:")
            transfers_by_class = summary.get("transfers_by_class", {})
            
            for class_name, class_stats in transfers_by_class.items():
                count = class_stats.get("count", 0)
                total_objects = class_stats.get("total_objects", 0)
                
                if "ok" in class_stats and "no" in class_stats:
                    # Classes com OK/NO (FIFA, Simbolo, String)
                    ok_count = class_stats["ok"]
                    no_count = class_stats["no"]
                    text_content.append(f"  {class_name.upper()}:")
                    text_content.append(f"    - Transfers com detecção: {count}")
                    text_content.append(f"    - Total objetos: {total_objects} (OK: {ok_count}, NO: {no_count})")
                else:
                    # Classes sem OK/NO (Blackdot, Smudge)
                    text_content.append(f"  {class_name.upper()}:")
                    text_content.append(f"    - Transfers com detecção: {count}")
                    text_content.append(f"    - Total objetos: {total_objects}")
                text_content.append("")
            
            # Erro mais frequente
            most_frequent = summary.get("most_frequent_error", {})
            text_content.append("❌ Erro Mais Frequente:")
            text_content.append(f"  - Classe: {most_frequent.get('class', 'Nenhuma')} ({most_frequent.get('type', 'N/A')})")
            text_content.append(f"  - Quantidade: {most_frequent.get('count', 0)} detecções")
            text_content.append(f"  - Percentual: {most_frequent.get('percentage', 0.0):.1f}% do total de erros")
            text_content.append("")
            
            # Detalhes por transfer (primeiros 10)
            objects_per_transfer = summary.get("objects_per_transfer", [])
            if objects_per_transfer:
                text_content.append(f"📋 Resumo por Transfer (primeiros 10):")
                for transfer_obj in objects_per_transfer[:10]:
                    transfer_id = transfer_obj.get("transfer_id", 0)
                    text_content.append(f"  Transfer #{transfer_id}:")
                    text_content.append(f"    - Blackdot: {transfer_obj.get('blackdot', 0)}")
                    text_content.append(f"    - Smudge: {transfer_obj.get('smudge', 0)}")
                    fifa = transfer_obj.get('fifa', {})
                    simbolo = transfer_obj.get('simbolo', {})
                    string = transfer_obj.get('string', {})
                    text_content.append(f"    - FIFA: OK={fifa.get('ok', 0)}, NO={fifa.get('no', 0)}")
                    text_content.append(f"    - Simbolo: OK={simbolo.get('ok', 0)}, NO={simbolo.get('no', 0)}")
                    text_content.append(f"    - String: OK={string.get('ok', 0)}, NO={string.get('no', 0)}")
                    text_content.append("")
                
                if len(objects_per_transfer) > 10:
                    text_content.append(f"    ... e mais {len(objects_per_transfer) - 10} transfers")
        else:
            text_content.append("Nenhum transfer foi avaliado durante esta execução.")
        
        text_content.append("=" * 70)
        
        # Atualizar widget de texto
        self.summary_text.delete('1.0', tk.END)
        self.summary_text.insert('1.0', '\n'.join(text_content))
    
    def _start_update_loop(self):
        """Inicia loop de atualização da UI."""
        self._update_ui()
    
    def _update_ui(self):
        """Atualiza UI periodicamente - seguindo padrão do código de referência que funciona."""
        # Atualizar preview
        with self.frame_lock:
            if self.current_frame is not None:
                frame = self.current_frame.copy()
                
                # Seguir exatamente o padrão do código de referência para evitar deslocamento
                h, w = frame.shape[:2]
                label_width = self.preview_label.winfo_width()
                label_height = self.preview_label.winfo_height()
                
                if label_width > 10 and label_height > 10:  # Certifica que widget foi renderizado
                    # Calcula escala para manter aspect ratio (igual código de referência)
                    scale_w = label_width / w
                    scale_h = label_height / h
                    scale = min(scale_w, scale_h)  # Usa menor escala para manter proporção
                    
                    new_width = int(w * scale)
                    new_height = int(h * scale)
                    
                    # Redimensiona com interpolação de alta qualidade (igual código de referência)
                    if new_width > 10 and new_height > 10:
                        frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
                    
                    # Se a imagem não preencher completamente, adiciona padding preto centralizado
                    # (igual código de referência - isso evita deslocamento)
                    if new_width != label_width or new_height != label_height:
                        # Cria imagem preta do tamanho do label
                        display_frame = np.zeros((label_height, label_width, 3), dtype=np.uint8)
                        
                        # Centraliza a imagem redimensionada (igual código de referência)
                        y_offset = (label_height - new_height) // 2
                        x_offset = (label_width - new_width) // 2
                        
                        display_frame[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = frame
                        frame = display_frame
                
                # Converter para PhotoImage
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                imgtk = ImageTk.PhotoImage(image=img)
                
                self.preview_label.configure(image=imgtk, text="")
                self.preview_label.image = imgtk  # Manter referência
        
        # Atualizar estatísticas
        self.fps_label.config(text=f"{self.stats.get('fps', 0):.1f}")
        self.capture_fps_label.config(text=f"{self.stats.get('capture_fps', 0):.1f}")
        self.infer_label.config(text=f"{self.stats.get('inference_ms', 0):.1f} ms")
        self.smudge_label.config(text=str(self.stats.get('smudge', 0)))
        self.simbolos_label.config(text=str(self.stats.get('simbolos', 0)))
        self.blackdot_label.config(text=str(self.stats.get('blackdot', 0)))
        self.transfer_label.config(text=str(self.stats.get('transfer_count', 0)))
        self.avg_smudge_label.config(text=f"{self.stats.get('avg_smudge', 0):.1f}")
        self.avg_simbolos_label.config(text=f"{self.stats.get('avg_simbolos', 0):.1f}")
        self.avg_blackdot_label.config(text=f"{self.stats.get('avg_blackdot', 0):.1f}")
        
        # Atualizar estatísticas de transfer
        self.evaluated_label.config(text=str(self.stats.get('total_evaluated', 0)))
        self.approved_label.config(text=str(self.stats.get('total_approved', 0)))
        self.rejected_label.config(text=str(self.stats.get('total_rejected', 0)))
        self.approval_rate_label.config(text=f"{self.stats.get('approval_rate', 0):.1f}%")
        
        # Atualizar estatísticas de classes detectadas médias
        self.avg_smudge_detected_label.config(text=f"{self.stats.get('avg_smudge_detected', 0):.1f}%")
        self.avg_simbolos_detected_label.config(text=f"{self.stats.get('avg_simbolos_detected', 0):.1f}%")
        self.avg_blackdot_detected_label.config(text=f"{self.stats.get('avg_blackdot_detected', 0):.1f}%")
        
        # Agendar próxima atualização
        self.root.after(33, self._update_ui)  # ~30 FPS na UI
    
    def set_status(self, message: str, color: str = "black"):
        """Define mensagem de status."""
        self.status_label.config(text=message, foreground=color)
    
    def minimize_window(self):
        """Minimiza a janela."""
        self.root.iconify()
    
    def on_closing(self):
        """Handler de fechamento da janela."""
        if self.running and self.on_stop:
            self.on_stop()
        self.root.quit()
        self.root.destroy()
    
    def run(self):
        """Inicia o loop principal da UI."""
        self.root.mainloop()
    
    def get_thresholds(self) -> Dict[str, float]:
        """Retorna valores atuais dos thresholds."""
        return {
            "smudge_conf": self.smudge_conf_var.get(),
            "simbolo_conf": self.simbolos_conf_var.get(),
            "blackdot_conf": self.blackdot_conf_var.get(),
            "roi_conf": self.roi_conf_var.get()
        }
    
    def get_camera_params(self) -> Dict[str, Any]:
        """Retorna valores atuais dos parâmetros da câmera."""
        # Extrair resolução
        resolution_str = self.cam_resolution_var.get().split(' ')[0]  # "1280x720"
        width, height = map(int, resolution_str.split('x'))
        
        return {
            "fps": int(self.cam_fps_var.get()),
            "exposure": int(self.cam_exposure_var.get()),
            "gain": self.cam_gain_var.get(),
            "balance_white_auto": self.cam_balance_var.get(),
            "resolution": (width, height)
        }
    
    def is_paused(self) -> bool:
        """Retorna se está pausado."""
        return self.paused
    
    def is_recording(self) -> bool:
        """Retorna se está gravando."""
        return self.recording
    
    def _on_model_toggle(self, model_name: str, enabled: bool):
        """Handler de toggle de modelo."""
        if self.on_model_toggle:
            self.on_model_toggle(model_name, enabled)
    
    def _enable_all_models(self):
        """Ativa todos os modelos."""
        for model_name, var in self.model_vars.items():
            var.set(True)
            self._on_model_toggle(model_name, True)
    
    def _disable_all_models(self):
        """Desativa todos os modelos."""
        for model_name, var in self.model_vars.items():
            var.set(False)
            self._on_model_toggle(model_name, False)
    
    def get_model_status(self) -> dict:
        """Retorna status atual dos modelos."""
        return {name: var.get() for name, var in self.model_vars.items()}
    
    def update_statistics(self, stats: dict):
        """Atualiza o sumário de estatísticas."""
        try:
            # Transfers processados
            transfers = stats.get("transfer_count", 0)
            self.stats_labels["transfers"].config(text=f"Transfers: {transfers}")
            
            # ROI Confidence
            roi_conf = stats.get("roi_confidence", 0.0)
            if roi_conf > 0:
                self.stats_labels["roi_conf"].config(text=f"ROI Conf: {roi_conf:.3f}")
            else:
                self.stats_labels["roi_conf"].config(text="ROI Conf: -")
            
            # Classes detectadas e quantidades
            classes_detected = []
            quantities = []
            
            if stats.get("smudge", 0) > 0:
                classes_detected.append("Smudge")
                quantities.append(f"Smudge: {stats['smudge']}")
                
                if stats.get("simbolos", 0) > 0:
                    classes_detected.append("Símbolo")
                    quantities.append(f"Símbolo: {stats['simbolos']}")
                
                if stats.get("blackdot", 0) > 0:
                    classes_detected.append("BlackDot")
                    quantities.append(f"BlackDot: {stats['blackdot']}")
            
            # Atualizar labels das classes detectadas
            if classes_detected:
                self.stats_labels["classes"].config(text=f"Classes: {', '.join(classes_detected)}")
                self.stats_labels["quantities"].config(text=f"Quantidades: {' | '.join(quantities)}")
            else:
                self.stats_labels["classes"].config(text="Classes: Nenhuma")
                self.stats_labels["quantities"].config(text="Quantidades: -")
            
            # Classe predominante estabilizada (média móvel das últimas 10 frames)
            predominant_class = stats.get("predominant_class", "Nenhuma")
            predominant_confidence = stats.get("predominant_class_confidence", 0.0)
            
            # Atualizar label da classe predominante
            if predominant_class != "Nenhuma":
                confidence_percent = predominant_confidence * 100
                
                # Exibir detalhes das classes OK/NO se disponíveis
                ok_no_details = []
                if stats.get("smudge_ok", 0) > 0 or stats.get("smudge_no", 0) > 0:
                    ok_no_details.append(f"Smudge: {stats.get('smudge_ok', 0)}OK/{stats.get('smudge_no', 0)}NO")
                if stats.get("simbolo_ok", 0) > 0 or stats.get("simbolo_no", 0) > 0:
                    ok_no_details.append(f"Símbolo: {stats.get('simbolo_ok', 0)}OK/{stats.get('simbolo_no', 0)}NO")
                if stats.get("blackdot_ok", 0) > 0 or stats.get("blackdot_no", 0) > 0:
                    ok_no_details.append(f"BlackDot: {stats.get('blackdot_ok', 0)}OK/{stats.get('blackdot_no', 0)}NO")
                # R não existe mais no modelo best.pt (removido - agora são apenas FIFA, Simbolo, String)
                
                if ok_no_details:
                    details_text = " | ".join(ok_no_details)
                    self.stats_labels["predominant_class"].config(
                        text=f"🎯 Predominante: {predominant_class} ({confidence_percent:.1f}%) - {details_text}",
                        foreground="#2E7D32"
                    )
                else:
                    self.stats_labels["predominant_class"].config(
                        text=f"🎯 Predominante: {predominant_class} ({confidence_percent:.1f}%)",
                        foreground="#2E7D32"
                    )
            else:
                self.stats_labels["predominant_class"].config(
                    text="🎯 Predominante: Nenhuma",
                    foreground="#666666"  # Cinza para indicar ausência
                )
            
            # Atualizar janela detalhada se estiver aberta
            self._update_detailed_stats_window(stats)
                
        except Exception as e:
            print(f"Erro ao atualizar estatísticas: {e}")
    
    def _update_detailed_stats_window(self, stats: dict):
        """Atualiza a janela de estatísticas detalhadas."""
        if hasattr(self, 'detailed_labels') and self.detailed_labels:
            try:
                # Transfers
                transfers = stats.get("transfer_count", 0)
                self.detailed_labels["transfers"].config(text=f"Transfers Processados: {transfers}")
                
                # ROI Confidence
                roi_conf = stats.get("roi_confidence", 0.0)
                if roi_conf > 0:
                    self.detailed_labels["roi_conf"].config(text=f"ROI Confidence: {roi_conf:.3f}")
                else:
                    self.detailed_labels["roi_conf"].config(text="ROI Confidence: -")
                
                # FPS
                fps = stats.get("capture_fps", 0.0)
                self.detailed_labels["fps"].config(text=f"FPS Câmera: {fps:.1f}")
                
                # Detecções por classe
                self.detailed_labels["smudge"].config(text=f"Smudge: {stats.get('smudge', 0)}")
                self.detailed_labels["simbolos"].config(text=f"Símbolos: {stats.get('simbolos', 0)}")
                self.detailed_labels["blackdot"].config(text=f"BlackDot: {stats.get('blackdot', 0)}")
                
                # Performance
                inference_time = stats.get("inference_time_ms", 0.0)
                self.detailed_labels["inference_time"].config(text=f"Tempo de Inferência: {inference_time:.1f}ms")
                
                device = stats.get("device", "cuda:0")
                self.detailed_labels["device"].config(text=f"Device: {device}")
                
            except Exception as e:
                print(f"Erro ao atualizar janela detalhada: {e}")
    
    def save_controls_state(self) -> dict:
        """Salva o estado atual dos controles."""
        return {
            "camera": {
                "fps": self.cam_fps_var.get(),
                "exposure": self.cam_exposure_var.get(),
                "gain": self.cam_gain_var.get(),
                "balance": self.cam_balance_var.get(),
                "resolution": self.cam_resolution_var.get()
            },
            "thresholds": {
                "smudge": self.smudge_conf_var.get(),
                "simbolos": self.simbolos_conf_var.get(),
                "blackdot": self.blackdot_conf_var.get()
            },
            "models": {
                name: var.get() for name, var in self.model_vars.items()
            },
            "focus": {
                "focus": self.focus_var.get(),
                "auto_focus": self.auto_focus_var.get()
            }
        }
    
    def load_controls_state(self, state: dict):
        """Carrega o estado dos controles."""
        try:
            # Carregar configurações da câmera
            if "camera" in state:
                cam_state = state["camera"]
                self.cam_fps_var.set(cam_state.get("fps", 4))
                self.cam_exposure_var.set(cam_state.get("exposure", 5000))
                self.cam_gain_var.set(cam_state.get("gain", 0))
                self.cam_balance_var.set(cam_state.get("balance", "Continuous"))
                self.cam_resolution_var.set(cam_state.get("resolution", "1280x720 (HD)"))
            
            # Carregar thresholds (converter de 0.0-1.0 para 0-100)
            if "thresholds" in state:
                thresh_state = state["thresholds"]
                self.smudge_conf_var.set(thresh_state.get("smudge", 0.95) * 100)
                self.simbolos_conf_var.set(thresh_state.get("simbolo_conf", 0.5) * 100)
                self.blackdot_conf_var.set(thresh_state.get("blackdot", 0.1) * 100)
            
            # Carregar status dos modelos
            if "models" in state:
                model_state = state["models"]
                for name, enabled in model_state.items():
                    if name in self.model_vars:
                        self.model_vars[name].set(enabled)
            
            # Carregar controles de foco
            if "focus" in state:
                focus_state = state["focus"]
                self.focus_var.set(focus_state.get("focus", 0))
                self.auto_focus_var.set(focus_state.get("auto_focus", False))
                self.focus_label.config(text=str(self.focus_var.get()))
            
            print("✓ Configurações dos controles carregadas com sucesso")
            
        except Exception as e:
            print(f"Erro ao carregar configurações dos controles: {e}")
    
    def _on_focus_change(self, _=None):
        """Função removida - sistema usa ajuste automático."""
        pass
    
    def _on_sharpness_change(self, _=None):
        """Função removida - sistema usa ajuste automático."""
        pass
    
    def _on_auto_focus_change(self):
        """Função removida - sistema usa ajuste automático."""
        pass
    
    def _on_auto_sharpness_change(self):
        """Função removida - sistema usa ajuste automático."""
        pass
    
    def _focus_decrease(self):
        """Função removida - sistema usa ajuste automático."""
        pass
    
    def _focus_increase(self):
        """Função removida - sistema usa ajuste automático."""
        pass
    
    def _auto_focus_trigger(self):
        """Função removida - sistema usa ajuste automático."""
        pass
    
    def _auto_sharpness_trigger(self):
        """Dispara auto-nitidez."""
        if self.on_focus_change:
            self.on_focus_change({"auto_sharpness_trigger": True})
    
    def _auto_both_trigger(self):
        """Dispara auto-foco e auto-nitidez simultaneamente."""
        if self.on_focus_change:
            self.on_focus_change({
                "auto_focus_trigger": True,
                "auto_sharpness_trigger": True
            })
    
    def _beep_focus(self, value):
        """Beep sonoro para ajuste de foco com sensibilidade."""
        try:
            import winsound
            # Frequência baseada no valor (200-1000 Hz) - faixa mais ampla
            frequency = int(200 + (value * 8))
            
            # Duração baseada na proximidade de valores "ótimos" (25%, 50%, 75%)
            optimal_points = [25, 50, 75]
            distances = [abs(value - point) for point in optimal_points]
            min_distance = min(distances)
            
            # Duração varia com proximidade do ponto ótimo
            if min_distance <= 5:  # Muito próximo de um ponto ótimo
                duration = 100  # Beep mais longo
                # Frequência ligeiramente diferente para pontos ótimos
                frequency = int(frequency * 1.1)
            elif min_distance <= 10:  # Próximo de um ponto ótimo
                duration = 75
            else:  # Longe dos pontos ótimos
                duration = 50
                
            # Volume (intensidade) baseado na mudança de valor
            if hasattr(self, '_last_focus_value'):
                change = abs(value - self._last_focus_value)
                if change > 10:  # Mudança significativa
                    # Beep duplo para mudanças grandes
                    winsound.Beep(frequency, duration)
                    winsound.Beep(int(frequency * 0.8), duration // 2)
                else:
                    winsound.Beep(frequency, duration)
            else:
                winsound.Beep(frequency, duration)
                
            # Armazenar valor anterior para próxima comparação
            self._last_focus_value = value
            
        except ImportError:
            # Fallback para sistemas sem winsound
            print(f"\a")  # Beep do sistema
    
    def _beep_sharpness(self, value):
        """Beep sonoro para ajuste de nitidez com sensibilidade."""
        try:
            import winsound
            # Frequência diferente para nitidez (400-1200 Hz) - faixa mais ampla
            frequency = int(400 + (value * 8))
            
            # Duração baseada na proximidade de valores "ótimos" (30%, 60%, 90%)
            optimal_points = [30, 60, 90]
            distances = [abs(value - point) for point in optimal_points]
            min_distance = min(distances)
            
            # Duração varia com proximidade do ponto ótimo
            if min_distance <= 5:  # Muito próximo de um ponto ótimo
                duration = 80  # Beep mais longo
                # Frequência ligeiramente diferente para pontos ótimos
                frequency = int(frequency * 1.05)
            elif min_distance <= 10:  # Próximo de um ponto ótimo
                duration = 60
            else:  # Longe dos pontos ótimos
                duration = 40
                
            # Volume (intensidade) baseado na mudança de valor
            if hasattr(self, '_last_sharpness_value'):
                change = abs(value - self._last_sharpness_value)
                if change > 8:  # Mudança significativa
                    # Beep duplo para mudanças grandes
                    winsound.Beep(frequency, duration)
                    winsound.Beep(int(frequency * 0.9), duration // 2)
                else:
                    winsound.Beep(frequency, duration)
            else:
                winsound.Beep(frequency, duration)
                
            # Armazenar valor anterior para próxima comparação
            self._last_sharpness_value = value
            
        except ImportError:
            print(f"\a")  # Beep do sistema
    
    def _beep_test(self):
        """Teste do beep sonoro."""
        try:
            import winsound
            # Sequência de beeps para teste
            winsound.Beep(440, 100)  # Lá
            self.root.after(150, lambda: winsound.Beep(554, 100))  # Dó#
            self.root.after(300, lambda: winsound.Beep(659, 100))  # Mi
        except ImportError:
            print(f"\a\a\a")  # Três beeps do sistema
    
    def _toggle_continuous_sound(self):
        """Ativa/desativa som contínuo."""
        self.continuous_sound_active = not self.continuous_sound_active
        
        if self.continuous_sound_active:
            # Iniciar som contínuo
            self._start_continuous_sound()
            # Atualizar botão visualmente
            for widget in self.root.winfo_children():
                if isinstance(widget, ttk.Frame):
                    for child in widget.winfo_children():
                        if isinstance(child, ttk.Frame):
                            for btn in child.winfo_children():
                                if isinstance(btn, ttk.Button) and btn.cget('text') == '🔁':
                                    btn.config(text='🔴')  # Indicar ativo
        else:
            # Parar som contínuo
            self._stop_continuous_sound()
            # Atualizar botão visualmente
            for widget in self.root.winfo_children():
                if isinstance(widget, ttk.Frame):
                    for child in widget.winfo_children():
                        if isinstance(child, ttk.Frame):
                            for btn in child.winfo_children():
                                if isinstance(btn, ttk.Button) and btn.cget('text') == '🔴':
                                    btn.config(text='🔁')  # Indicar inativo
    
    def _start_continuous_sound(self):
        """Inicia som contínuo baseado nos valores atuais."""
        if self.continuous_sound_active:
            # Som baseado no foco atual
            focus_value = self.focus_var.get()
            self._continuous_beep_focus(focus_value)
            
            # Agendar próximo som
            self.sound_timer = self.root.after(200, self._start_continuous_sound)
    
    def _stop_continuous_sound(self):
        """Para o som contínuo."""
        if self.sound_timer:
            self.root.after_cancel(self.sound_timer)
            self.sound_timer = None
    
    def _continuous_beep_focus(self, value):
        """Beep contínuo para foco."""
        try:
            import winsound
            # Frequência baseada no valor (200-1000 Hz)
            frequency = int(200 + (value * 8))
            
            # Duração mais curta para som contínuo
            duration = 30
            
            # Beep mais suave para som contínuo
            winsound.Beep(frequency, duration)
            
        except ImportError:
            print(f"\a")  # Beep do sistema
    
    def _open_stats_window(self):
        """Abre janela de estatísticas detalhadas."""
        if not hasattr(self, 'stats_window') or not self.stats_window.winfo_exists():
            self.stats_window = tk.Toplevel(self.root)
            self.stats_window.title("📊 Estatísticas Detalhadas")
            self.stats_window.geometry("700x600")  # Maior para acomodar o sumário
            self.stats_window.resizable(True, True)
            
            # Frame principal
            main_frame = ttk.Frame(self.stats_window, padding="10")
            main_frame.pack(fill=tk.BOTH, expand=True)
            
            # Estatísticas gerais
            general_frame = ttk.LabelFrame(main_frame, text="📈 Estatísticas Gerais", padding="10")
            general_frame.pack(fill=tk.X, pady=5)
            
            self.detailed_labels = {}
            
            # Transfers
            self.detailed_labels["transfers"] = ttk.Label(general_frame, text="Transfers Processados: 0", font=("Arial", 11, "bold"))
            self.detailed_labels["transfers"].pack(anchor=tk.W)
            
            # ROI Confidence
            self.detailed_labels["roi_conf"] = ttk.Label(general_frame, text="ROI Confidence: -", font=("Arial", 11))
            self.detailed_labels["roi_conf"].pack(anchor=tk.W)
            
            # Performance
            self.detailed_labels["fps"] = ttk.Label(general_frame, text="FPS Câmera: -", font=("Arial", 11))
            self.detailed_labels["fps"].pack(anchor=tk.W)
            
            self.detailed_labels["capture_fps"] = ttk.Label(general_frame, text="FPS Captura: -", font=("Arial", 11))
            self.detailed_labels["capture_fps"].pack(anchor=tk.W)
            
            # Estatísticas por classe
            classes_frame = ttk.LabelFrame(main_frame, text="🎯 Detecções por Classe", padding="10")
            classes_frame.pack(fill=tk.X, pady=5)
            
            self.detailed_labels["smudge"] = ttk.Label(classes_frame, text="Smudge: 0", font=("Arial", 11))
            self.detailed_labels["smudge"].pack(anchor=tk.W)
            
            self.detailed_labels["simbolos"] = ttk.Label(classes_frame, text="Símbolos: 0", font=("Arial", 11))
            self.detailed_labels["simbolos"].pack(anchor=tk.W)
            
            self.detailed_labels["blackdot"] = ttk.Label(classes_frame, text="BlackDot: 0", font=("Arial", 11))
            self.detailed_labels["blackdot"].pack(anchor=tk.W)
            
            # Estatísticas de performance
            perf_frame = ttk.LabelFrame(main_frame, text="⚡ Performance", padding="10")
            perf_frame.pack(fill=tk.X, pady=5)
            
            self.detailed_labels["inference_time"] = ttk.Label(perf_frame, text="Tempo de Inferência: -", font=("Arial", 11))
            self.detailed_labels["inference_time"].pack(anchor=tk.W)
            
            self.detailed_labels["device"] = ttk.Label(perf_frame, text="Device: -", font=("Arial", 11))
            self.detailed_labels["device"].pack(anchor=tk.W)
            
            # Estatísticas de transfer
            transfer_frame = ttk.LabelFrame(main_frame, text="🔄 Estatísticas de Transfer", padding="10")
            transfer_frame.pack(fill=tk.X, pady=5)
            
            self.detailed_labels["transfer_count"] = ttk.Label(transfer_frame, text="Transfer #: 0", font=("Arial", 11))
            self.detailed_labels["transfer_count"].pack(anchor=tk.W)
            
            self.detailed_labels["total_evaluated"] = ttk.Label(transfer_frame, text="Total Avaliados: 0", font=("Arial", 11))
            self.detailed_labels["total_evaluated"].pack(anchor=tk.W)
            
            self.detailed_labels["total_approved"] = ttk.Label(transfer_frame, text="Total Aprovados: 0", font=("Arial", 11))
            self.detailed_labels["total_approved"].pack(anchor=tk.W)
            
            self.detailed_labels["total_rejected"] = ttk.Label(transfer_frame, text="Total Reprovados: 0", font=("Arial", 11))
            self.detailed_labels["total_rejected"].pack(anchor=tk.W)
            
            self.detailed_labels["approval_rate"] = ttk.Label(transfer_frame, text="Taxa Aprovação: 0.0%", font=("Arial", 11))
            self.detailed_labels["approval_rate"].pack(anchor=tk.W)
            
            # Médias por transfer
            avg_frame = ttk.LabelFrame(main_frame, text="📊 Médias por Transfer", padding="10")
            avg_frame.pack(fill=tk.X, pady=5)
            
            self.detailed_labels["avg_smudge"] = ttk.Label(avg_frame, text="Smudge Médio: 0.0", font=("Arial", 11))
            self.detailed_labels["avg_smudge"].pack(anchor=tk.W)
            
            self.detailed_labels["avg_simbolos"] = ttk.Label(avg_frame, text="Símbolo Médio: 0.0", font=("Arial", 11))
            self.detailed_labels["avg_simbolos"].pack(anchor=tk.W)
            
            self.detailed_labels["avg_blackdot"] = ttk.Label(avg_frame, text="BlackDot Médio: 0.0", font=("Arial", 11))
            self.detailed_labels["avg_blackdot"].pack(anchor=tk.W)
            
            # Classes detectadas médias
            detected_frame = ttk.LabelFrame(main_frame, text="🎯 Classes Detectadas Médias", padding="10")
            detected_frame.pack(fill=tk.X, pady=5)
            
            self.detailed_labels["avg_smudge_detected"] = ttk.Label(detected_frame, text="Smudge Detectado: 0.0%", font=("Arial", 11))
            self.detailed_labels["avg_smudge_detected"].pack(anchor=tk.W)
            
            self.detailed_labels["avg_simbolos_detected"] = ttk.Label(detected_frame, text="Símbolos Detectado: 0.0%", font=("Arial", 11))
            self.detailed_labels["avg_simbolos_detected"].pack(anchor=tk.W)
            
            self.detailed_labels["avg_blackdot_detected"] = ttk.Label(detected_frame, text="BlackDot Detectado: 0.0%", font=("Arial", 11))
            self.detailed_labels["avg_blackdot_detected"].pack(anchor=tk.W)
            
            # Sumário final de estatísticas
            summary_frame = ttk.LabelFrame(main_frame, text="📊 Sumário Final de Processamento", padding="10")
            summary_frame.pack(fill=tk.BOTH, expand=True, pady=5)
            
            # Scrollable text widget para o sumário
            summary_scroll = tk.Scrollbar(summary_frame)
            summary_scroll.pack(side=tk.RIGHT, fill=tk.Y)
            
            self.summary_text = tk.Text(summary_frame, height=10, font=("Courier", 9), 
                                       wrap=tk.WORD, yscrollcommand=summary_scroll.set)
            self.summary_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            summary_scroll.config(command=self.summary_text.yview)
            
            self.detailed_labels["summary_text"] = self.summary_text
            
            # Botão de atualização
            ttk.Button(main_frame, text="🔄 Atualizar", command=self._update_detailed_stats).pack(pady=10)
            
            # Inicializar com dados atuais
            self._update_detailed_stats()
            
            # Se já temos sumário, atualizar exibição
            if self.statistics_summary:
                self._update_summary_display()
        else:
            self.stats_window.lift()
            # Atualizar se temos sumário
            if hasattr(self, 'summary_text') and self.statistics_summary:
                self._update_summary_display()
    
    def _update_detailed_stats(self):
        """Atualiza estatísticas detalhadas."""
        if hasattr(self, 'detailed_labels'):
            try:
                # Atualizar estatísticas básicas
                self.detailed_labels["fps"].config(text=f"FPS: {self.stats.get('fps', 0):.1f}")
                self.detailed_labels["capture_fps"].config(text=f"FPS Captura: {self.stats.get('capture_fps', 0):.1f}")
                self.detailed_labels["inference_time"].config(text=f"Tempo de Inferência: {self.stats.get('inference_ms', 0):.1f} ms")
                self.detailed_labels["device"].config(text=f"Device: {self.stats.get('device', 'N/A')}")
                
                # Atualizar estatísticas de detecção
                self.detailed_labels["smudge"].config(text=f"FIFA: {self.stats.get('smudge', 0)}")
                self.detailed_labels["simbolos"].config(text=f"Símbolos: {self.stats.get('simbolos', 0)}")
                self.detailed_labels["blackdot"].config(text=f"BlackDot: {self.stats.get('blackdot', 0)}")
                
                # Atualizar estatísticas de transfer
                self.detailed_labels["transfer_count"].config(text=f"Transfer #: {self.stats.get('transfer_count', 0)}")
                self.detailed_labels["total_evaluated"].config(text=f"Total Avaliados: {self.stats.get('total_evaluated', 0)}")
                self.detailed_labels["total_approved"].config(text=f"Total Aprovados: {self.stats.get('total_approved', 0)}")
                self.detailed_labels["total_rejected"].config(text=f"Total Reprovados: {self.stats.get('total_rejected', 0)}")
                self.detailed_labels["approval_rate"].config(text=f"Taxa Aprovação: {self.stats.get('approval_rate', 0):.1f}%")
                
                # Atualizar médias por transfer
                self.detailed_labels["avg_smudge"].config(text=f"Smudge Médio: {self.stats.get('avg_smudge', 0):.1f}")
                self.detailed_labels["avg_simbolos"].config(text=f"Símbolo Médio: {self.stats.get('avg_simbolos', 0):.1f}")
                self.detailed_labels["avg_blackdot"].config(text=f"BlackDot Médio: {self.stats.get('avg_blackdot', 0):.1f}")
                
                # Atualizar classes detectadas médias
                self.detailed_labels["avg_smudge_detected"].config(text=f"Smudge Detectado: {self.stats.get('avg_smudge_detected', 0):.1f}%")
                self.detailed_labels["avg_simbolos_detected"].config(text=f"Símbolo Detectado: {self.stats.get('avg_simbolos_detected', 0):.1f}%")
                self.detailed_labels["avg_blackdot_detected"].config(text=f"BlackDot Detectado: {self.stats.get('avg_blackdot_detected', 0):.1f}%")
                
                # Atualizar sumário se disponível
                self._update_summary_display()
                
                print("✅ Estatísticas atualizadas com sucesso!")
                
            except Exception as e:
                print(f"❌ Erro ao atualizar estatísticas: {e}")
                import traceback
                traceback.print_exc()

