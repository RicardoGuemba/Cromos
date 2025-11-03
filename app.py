"""
Aplicação principal do Sistema de Detecção YOLO com câmera Basler.
Integra captura, inferência e UI em pipeline multi-thread.
"""

import os
import sys
import logging
import threading
import time
import queue
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

import yaml
import cv2
import numpy as np
import torch

from camera_basler import BaslerCamera
from infer import YOLODetector
from ui_v2 import YOLODetectionUI  # NOVA UI V2
from config_manager import ConfigManager


# Configurar logging
def setup_logging():
    """Configura sistema de logging."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    log_file = log_dir / f"yolo_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)


class YOLODetectionApp:
    """Aplicação principal de detecção YOLO."""
    
    def __init__(self, config_path: str = "config/app.yaml"):
        """Inicializa a aplicação."""
        self.logger = setup_logging()
        self.logger.info("="*60)
        self.logger.info("Iniciando YOLO Detection System - Basler USB3 Vision")
        self.logger.info("="*60)
        
        # Carregar configuração
        self.config = self._load_config(config_path)
        
        # Verificar CUDA e otimizar para RTX 3050
        self._check_cuda()
        self._optimize_for_rtx3050()
        
        # Componentes
        self.camera: Optional[BaslerCamera] = None
        self.detector: Optional[YOLODetector] = None
        self.ui: Optional[YOLODetectionUI] = None
        
        # Threads
        self.inference_thread: Optional[threading.Thread] = None
        self.writer_thread: Optional[threading.Thread] = None
        
        # Controle
        self.running = False
        self.recording = False
        self.video_writer: Optional[cv2.VideoWriter] = None
        
        # Buffers
        self.frame_queue = queue.Queue(maxsize=8)
        self.processed_queue = queue.Queue(maxsize=4)
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0.0
        
    def _load_config(self, config_path: str) -> dict:
        """Carrega arquivo de configuração YAML com persistência."""
        # Inicializar gerenciador de configurações
        self.config_manager = ConfigManager(config_path)
        
        # Carregar configuração base
        config = self.config_manager.load_config()
        if not config:
            self.logger.error("Erro ao carregar configuração base")
            self.logger.info("Usando configuração padrão")
            return self._get_default_config()
        
        # Carregar configurações salvas e mesclar
        saved_settings = self.config_manager.load_settings()
        if saved_settings:
            config = self.config_manager.merge_settings(config, saved_settings)
            self.logger.info("✓ Configurações salvas restauradas")
        else:
            self.logger.info("✓ Usando configuração padrão")
        
        return config
    
    def _get_default_config(self) -> dict:
        """Retorna configuração padrão."""
        return {
            "camera": {
                "fps_target": 4,  # Faixa permitida: 1-200 FPS (faixa nominal completa)
                "pixel_format": "Mono8",
                "width": 1280,
                "height": 720,
                "exposure_time": 10000,  # 10ms - valor mais usual
                "gain": 0,
                "timeout_ms": 50,
                "balance_white_auto": "Off"  # Off, Once, ou Continuous
            },
            "inference": {
                "imgsz": 640,
                "max_det": 100,
                "device": "cuda"
            },
            "models": {
                "seg": "models/Crop_Fifa_best.pt",
                "smudge": "models/best_smudge.pt",
                "simbolos": "models/best.pt",  # Modelo com 6 classes (FIFA, Simbolo, String - cada com OK/NO)
                "blackdot": "models/best_blackdot.pt"
            },
            "recording": {
                "codec": "mp4v",  # MPEG-4 - padrão MP4 funcionando
                "fps": 4,
                "output_dir": "recordings"
            },
            "roi": {
                "conf": 0.5,  # Default: 0.5 (50%)
                "iou": 0.45,
                "min_pixels": 2000
            },
            "thresholds": {
                "smudge_conf": 0.5,  # Default: 0.5 (50%)
                "smudge_iou": 0.45,
                "simbolo_conf": 0.5,  # Default: 0.5 (50%)
                "simbolo_iou": 0.45,
                "blackdot_conf": 0.5,  # Default: 0.5 (50%)
                "blackdot_iou": 0.45
            }
        }
    
    def _check_cuda(self):
        """Verifica disponibilidade CUDA."""
        self.logger.info("Verificando ambiente PyTorch...")
        self.logger.info(f"PyTorch version: {torch.__version__}")
        
        if torch.cuda.is_available():
            device = "cuda"
            self.logger.info(f"✓ CUDA disponível: {torch.cuda.get_device_name(device)}")
            self.logger.info(f"✓ CUDA version: {torch.version.cuda}")
            self.logger.info(f"✓ cuDNN version: {torch.backends.cudnn.version()}")
            
            # Informações de memória
            mem_total = torch.cuda.get_device_properties(device).total_memory / (1024**3)
            self.logger.info(f"✓ GPU Memory: {mem_total:.2f} GB")
        else:
            self.logger.warning("✗ CUDA não disponível - usando CPU")
    
    def _optimize_for_rtx3050(self):
        """Otimizações específicas para RTX 3050."""
        if torch.cuda.is_available():
            device = "cuda"
            gpu_name = torch.cuda.get_device_name(device).lower()
            if "rtx 3050" in gpu_name or "3050" in gpu_name:
                self.logger.info("🎯 Detectada RTX 3050 - Aplicando otimizações específicas...")
                
                # Configurar tamanho de imagem otimizado para RTX 3050
                if "inference" not in self.config:
                    self.config["inference"] = {}
                
                # Reduzir tamanho de imagem para melhor performance
                self.config["inference"]["imgsz"] = 512  # Reduzido de 640 para 512
                self.config["inference"]["max_det"] = 50  # Reduzido de 100 para 50
                
                # Configurar FPS da câmera otimizado
                if "camera" not in self.config:
                    self.config["camera"] = {}
                self.config["camera"]["fps_target"] = 4  # Valor padrão otimizado
                
                self.logger.info("✓ Configurações otimizadas para RTX 3050 aplicadas")
                self.logger.info(f"  - ImgSz: {self.config['inference']['imgsz']}")
                self.logger.info(f"  - Max Det: {self.config['inference']['max_det']}")
                self.logger.info(f"  - FPS Target: {self.config['camera']['fps_target']}")
            else:
                self.logger.info(f"GPU detectada: {torch.cuda.get_device_name("cuda")} - Usando configurações padrão")
    
    def _init_camera(self) -> bool:
        """Inicializa câmera Basler."""
        try:
            self.logger.info("Inicializando câmera Basler...")
            
            # Atualizar UI durante inicialização
            if self.ui:
                self.ui.root.update()
            
            cam_cfg = self.config.get("camera", {})
            self.camera = BaslerCamera(
                width=cam_cfg.get("width"),
                height=cam_cfg.get("height"),
                fps=cam_cfg.get("fps_target", 4),
                pixel_format=cam_cfg.get("pixel_format", "Mono8"),
                exposure_time=cam_cfg.get("exposure_time", 5000),
                gain=cam_cfg.get("gain", 0),
                timeout_ms=cam_cfg.get("timeout_ms", 50),
                balance_white_auto=cam_cfg.get("balance_white_auto", "Off")
            )
            
            # Tentar abrir com timeout para não travar
            if self.ui:
                self.ui.root.update()
            
            if not self.camera.open():
                self.logger.error("Falha ao abrir câmera Basler")
                if self.ui:
                    self.ui.root.update()
                return False
            
            if self.ui:
                self.ui.root.update()
            
            info = self.camera.get_info()
            self.logger.info(f"✓ Câmera inicializada: {info['width']}x{info['height']} @ {info['fps']} FPS")
            self.logger.info(f"✓ Balance White Auto: {info['balance_white_auto']}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Erro ao inicializar câmera: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            if self.ui:
                self.ui.root.update()
            return False
    
    def _init_detector(self) -> bool:
        """Inicializa sistema de detecção."""
        try:
            self.logger.info("Inicializando detector YOLO...")
            
            device = self.config.get("inference", {}).get("device", "cuda")
            self.detector = YOLODetector(self.config, device=device)
            
            if not self.detector.load_models():
                self.logger.error("Falha ao carregar modelos")
                return False
            
            # Carregar parâmetros salvos automaticamente
            if self.detector.load_parameters():
                self.logger.info("✓ Parâmetros anteriores carregados")
            else:
                self.logger.info("Usando parâmetros padrão")
            
            self.logger.info("Executando warm-up dos modelos...")
            self.detector.warmup()
            
            self.logger.info(f"✓ Detector inicializado no device: {self.detector.device}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Erro ao inicializar detector: {e}")
            return False
    
    def _init_ui(self):
        """Inicializa interface gráfica."""
        try:
            self.logger.info("Inicializando UI...")
            
            window_title = self.config.get("display", {}).get("window_title", 
                                                              "YOLO Detection System - Basler USB3 Vision")
            self.ui = YOLODetectionUI(window_title)
            
            # Conectar callbacks
            self.ui.on_start = self._on_ui_start
            self.ui.on_stop = self._on_ui_stop
            self.ui.on_record_toggle = self._on_ui_record_toggle
            self.ui.on_threshold_change = self._on_ui_threshold_change
            self.ui.on_camera_param_change = self._on_ui_camera_param_change  # NOVO
            self.ui.on_model_toggle = self._on_ui_model_toggle  # NOVO
            self.ui.on_focus_change = self._on_ui_focus_change  # NOVO
            self.ui.on_auto_camera_change = self._on_ui_auto_camera_change  # NOVO - Ajuste automático
            self.ui.on_class_change = self._on_ui_class_change  # NOVO - Controles de classes
            
            # Carregar configurações salvas dos controles
            if "ui_controls" in self.config:
                self.ui.load_controls_state(self.config["ui_controls"])
                self.logger.info("✓ Configurações dos controles carregadas")
            
            # Aplicar configurações salvas via ConfigManager
            if hasattr(self, 'config_manager'):
                try:
                    saved_settings = self.config_manager.load_settings()
                    if saved_settings:
                        self.config_manager.apply_ui_settings(self.ui, saved_settings)
                        self.logger.info("✓ Configurações da interface restauradas")
                except Exception as e:
                    self.logger.error(f"Erro ao restaurar configurações da UI: {e}")
            
            # Configurar thresholds iniciais
            thresholds_cfg = self.config.get("thresholds", {})
            self.ui.smudge_conf_var.set(thresholds_cfg.get("smudge_conf", 0.5))
            self.ui.simbolos_conf_var.set(thresholds_cfg.get("simbolo_conf", 0.5))
            self.ui.blackdot_conf_var.set(thresholds_cfg.get("blackdot_conf", 0.5))
            self.ui.roi_conf_var.set(self.config.get("roi", {}).get("conf", 0.5))
            
            # Configurar parâmetros iniciais da câmera
            cam_cfg = self.config.get("camera", {})
            self.ui.cam_fps_var.set(cam_cfg.get("fps_target", 4))
            self.ui.cam_exposure_var.set(cam_cfg.get("exposure_time", 5000))
            self.ui.cam_gain_var.set(cam_cfg.get("gain", 0))
            self.ui.cam_balance_var.set(cam_cfg.get("balance_white_auto", "Off"))
            
            # Configurar ajuste automático da câmera
            auto_cfg = self.config.get("camera_auto", {})
            self.ui.auto_exposure_var.set(auto_cfg.get("auto_exposure", False))
            self.ui.auto_gain_var.set(auto_cfg.get("auto_gain", False))
            
            self.logger.info("✓ UI inicializada")
            
        except Exception as e:
            self.logger.error(f"Erro ao inicializar UI: {e}")
            raise
    
    def _on_ui_start(self):
        """Callback: botão Iniciar."""
        self.logger.info("Iniciando captura e inferência...")
        
        # Ativar todos os modelos quando iniciar
        if self.detector:
            self.detector.enable_all_models()
            self.logger.info("✓ Todos os modelos ativados para início")
        
        self.start()
    
    def _on_ui_stop(self):
        """Callback: botão Parar."""
        self.logger.info("Parando captura e inferência...")
        self.stop()
    
    def _on_ui_record_toggle(self, recording: bool):
        """Callback: toggle de gravação."""
        if recording:
            self._start_recording()
        else:
            self._stop_recording()
    
    def _on_ui_threshold_change(self, thresholds: Dict[str, float]):
        """Callback: mudança de thresholds."""
        if self.detector:
            # Usar o novo método de atualização com persistência automática
            self.detector.update_thresholds(thresholds)
            self.logger.debug(f"Thresholds atualizados e salvos: {thresholds}")
            
            # Atualizar config local também
            self.config.setdefault("thresholds", {}).update({
                "smudge_conf": thresholds.get("smudge_conf", 0.5),
                "simbolo_conf": thresholds.get("simbolo_conf", 0.5),
                "blackdot_conf": thresholds.get("blackdot_conf", 0.5)
            })
            self.config.setdefault("roi", {})["conf"] = thresholds.get("roi_conf", 0.5)
    
    def _on_ui_camera_param_change(self, params: Dict[str, Any]):
        """Callback: mudança de parâmetros da câmera (com proteção)."""
        if not self.camera or not hasattr(self.camera, 'camera'):
            return
        
        # Balance White pode ser alterado sem sistema rodando
        if 'balance_white_auto' in params:
            mode = params['balance_white_auto']
            self.camera.set_balance_white_auto(mode)
            # Atualizar config
            self.config.setdefault("camera", {})["balance_white_auto"] = mode
            return
        
        if not self.running:
            self.logger.warning("Sistema não está rodando, ignorando mudança de parâmetros")
            return
            
        try:
            cam = self.camera.camera
            
            # Atualizar FPS (com validação) - Range expandido para faixa nominal
            if 'fps' in params and hasattr(cam, 'AcquisitionFrameRate'):
                fps = max(1, min(200, params['fps']))  # Limitar entre 1-200 (faixa nominal completa)
                cam.AcquisitionFrameRate.SetValue(fps)
                self.logger.info(f"✓ FPS atualizado: {fps}")
                # Atualizar config
                self.config.setdefault("camera", {})["fps_target"] = fps
            
            # Atualizar Exposição (com validação) - Range expandido para faixa nominal
            if 'exposure' in params and hasattr(cam, 'ExposureTime'):
                exposure = max(10, min(100000, params['exposure']))  # Limitar 10-100000µs (faixa nominal completa)
                cam.ExposureTime.SetValue(exposure)
                self.logger.info(f"✓ Exposição atualizada: {exposure} µs")
                # Atualizar config
                self.config.setdefault("camera", {})["exposure_time"] = exposure
            
            # Atualizar Ganho (com validação) - Range expandido para faixa nominal
            if 'gain' in params and hasattr(cam, 'Gain'):
                gain = max(0, min(48, params['gain']))  # Limitar 0-48dB (faixa nominal completa)
                cam.Gain.SetValue(gain)
                self.logger.info(f"✓ Ganho atualizado: {gain:.1f} dB")
                # Atualizar config
                self.config.setdefault("camera", {})["gain"] = gain
                
        except Exception as e:
            self.logger.error(f"Erro ao atualizar parâmetros da câmera: {e}")
    
    def _on_ui_model_toggle(self, model_name: str, enabled: bool):
        """Callback: mudança de status de modelo."""
        if self.detector:
            self.detector.set_model_enabled(model_name, enabled)
            self.logger.info(f"Modelo {model_name}: {'ATIVADO' if enabled else 'DESATIVADO'}")
    
    def _on_ui_focus_change(self, params: dict):
        """Callback: mudança de foco e nitidez."""
        try:
            if not self.camera or not hasattr(self.camera, 'camera'):
                return
            
            cam = self.camera.camera
            
            # Foco manual
            if 'focus' in params and hasattr(cam, 'FocusPos'):
                focus_value = params['focus']
                cam.FocusPos.SetValue(focus_value)
                self.logger.info(f"✓ Foco manual: {focus_value}%")
            
            # Nitidez manual
            if 'sharpness' in params and hasattr(cam, 'Sharpness'):
                sharpness_value = params['sharpness']
                cam.Sharpness.SetValue(sharpness_value)
                self.logger.info(f"✓ Nitidez manual: {sharpness_value}%")
            
            # Auto-foco
            if 'auto_focus' in params and hasattr(cam, 'FocusAuto'):
                auto_focus = params['auto_focus']
                if auto_focus:
                    cam.FocusAuto.SetValue("Continuous")
                    self.logger.info("✓ Auto-foco: ATIVADO")
                else:
                    cam.FocusAuto.SetValue("Off")
                    self.logger.info("✓ Auto-foco: DESATIVADO")
            
            # Auto-nitidez
            if 'auto_sharpness' in params and hasattr(cam, 'SharpnessAuto'):
                auto_sharpness = params['auto_sharpness']
                if auto_sharpness:
                    cam.SharpnessAuto.SetValue("Continuous")
                    self.logger.info("✓ Auto-nitidez: ATIVADO")
                else:
                    cam.SharpnessAuto.SetValue("Off")
                    self.logger.info("✓ Auto-nitidez: DESATIVADO")
            
            # Trigger de auto-foco
            if 'auto_focus_trigger' in params and hasattr(cam, 'FocusAuto'):
                cam.FocusAuto.SetValue("Once")
                self.logger.info("✓ Auto-foco: TRIGGER")
            
            # Trigger de auto-nitidez
            if 'auto_sharpness_trigger' in params and hasattr(cam, 'SharpnessAuto'):
                cam.SharpnessAuto.SetValue("Once")
                self.logger.info("✓ Auto-nitidez: TRIGGER")
            
            # Auto ajuste de ambos (foco e nitidez)
            if params.get('auto_focus_trigger') and params.get('auto_sharpness_trigger'):
                self.logger.info("✓ Auto ajuste completo: Foco + Nitidez")
            
        except Exception as e:
            self.logger.error(f"Erro ao atualizar foco/nitidez: {e}")
    
    def _on_ui_auto_camera_change(self, auto_params: Dict[str, bool]):
        """Callback: mudança de ajuste automático da câmera."""
        try:
            if not self.camera or not hasattr(self.camera, 'camera'):
                self.logger.warning("Câmera não disponível para ajuste automático")
                return
            
            cam = self.camera.camera
            
            # Ajuste automático de Exposição
            if auto_params.get('auto_exposure', False):
                try:
                    if hasattr(cam, 'ExposureAuto'):
                        cam.ExposureAuto.SetValue("Continuous")
                        self.logger.info("✓ Auto Exposição: ATIVADO")
                    else:
                        self.logger.warning("Auto Exposição não suportado nesta câmera")
                except Exception as e:
                    self.logger.warning(f"Erro ao ativar Auto Exposição: {e}")
            else:
                try:
                    if hasattr(cam, 'ExposureAuto'):
                        cam.ExposureAuto.SetValue("Off")
                        self.logger.info("✓ Auto Exposição: DESATIVADO")
                except Exception as e:
                    self.logger.warning(f"Erro ao desativar Auto Exposição: {e}")
            
            # Ajuste automático de Ganho
            if auto_params.get('auto_gain', False):
                try:
                    if hasattr(cam, 'GainAuto'):
                        cam.GainAuto.SetValue("Continuous")
                        self.logger.info("✓ Auto Ganho: ATIVADO")
                    else:
                        self.logger.warning("Auto Ganho não suportado nesta câmera")
                except Exception as e:
                    self.logger.warning(f"Erro ao ativar Auto Ganho: {e}")
            else:
                try:
                    if hasattr(cam, 'GainAuto'):
                        cam.GainAuto.SetValue("Off")
                        self.logger.info("✓ Auto Ganho: DESATIVADO")
                except Exception as e:
                    self.logger.warning(f"Erro ao desativar Auto Ganho: {e}")
            
            # Salvar configurações de ajuste automático
            self.config.setdefault("camera_auto", {}).update({
                "auto_exposure": auto_params.get('auto_exposure', False),
                "auto_gain": auto_params.get('auto_gain', False)
            })
            
        except Exception as e:
            self.logger.error(f"Erro ao configurar ajuste automático: {e}")
    
    def _on_ui_class_change(self, class_params: Dict[str, Any]):
        """Callback: mudança de classes específicas."""
        try:
            model_type = class_params.get('model_type')
            class_name = class_params.get('class_name')
            enabled = class_params.get('enabled', True)
            
            if model_type == "roi":
                self.logger.info(f"✓ Classe ROI '{class_name}': {'ATIVADA' if enabled else 'DESATIVADA'}")
            elif model_type == "simbolos":
                self.logger.info(f"✓ Classe Símbolos '{class_name}': {'ATIVADA' if enabled else 'DESATIVADA'}")
            
            # Salvar configurações de classes
            if "class_controls" not in self.config:
                self.config["class_controls"] = {}
            
            if model_type not in self.config["class_controls"]:
                self.config["class_controls"][model_type] = {}
            
            self.config["class_controls"][model_type][class_name] = enabled
            
        except Exception as e:
            self.logger.error(f"Erro ao configurar classes: {e}")
    
    def _start_recording(self):
        """Inicia gravação de vídeo com múltiplos codecs de fallback."""
        try:
            # Criar diretório de recordings
            rec_dir = Path(self.config.get("recording", {}).get("output_dir", "recordings"))
            rec_dir.mkdir(exist_ok=True)
            
            # Nome do arquivo
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = rec_dir / f"recording_{timestamp}.mp4"
            
            # Dimensões e FPS da câmera
            if self.camera:
                info = self.camera.get_info()
                width = info["width"]
                height = info["height"]
                # Usar FPS real da câmera (capture_fps se disponível) para velocidade natural no vídeo
                camera_fps = info.get("fps", self.config.get("camera", {}).get("fps_target", 4))
                # Se capture_fps estiver disponível e válido, usar ele (mais preciso)
                if hasattr(self.camera, 'capture_fps') and self.camera.capture_fps > 0:
                    camera_fps = self.camera.capture_fps
                rec_fps = max(1.0, min(camera_fps, 30.0))  # Limitar entre 1-30 FPS para compatibilidade
                self.logger.info(f"📹 FPS da câmera: {camera_fps:.2f}, FPS de gravação: {rec_fps:.2f} (velocidade natural)")
            else:
                width = 1280
                height = 720
                rec_fps = self.config.get("recording", {}).get("fps", 4)
            
            # Lista de codecs para tentar (em ordem de preferência)
            # Baseado no teste: mp4v é o melhor codec MP4 disponível
            codecs_to_try = [
                ("mp4v", "mp4v"),  # MPEG-4 - padrão MP4 funcionando
                ("MJPG", "MJPG"),  # Motion JPEG - fallback
                ("H264", "H264"),  # H.264 - pode não estar disponível
                ("XVID", "XVID"),  # XVID - alternativa
            ]
            
            self.video_writer = None
            successful_codec = None
            
            for codec_name, fourcc_str in codecs_to_try:
                try:
                    self.logger.info(f"Tentando codec: {codec_name}")
                    fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
                    
                    # Tentar criar VideoWriter
                    test_writer = cv2.VideoWriter(
                        str(output_file),
                        fourcc,
                        rec_fps,
                        (width, height)
                    )
                    
                    if test_writer.isOpened():
                        self.video_writer = test_writer
                        successful_codec = codec_name
                        self.logger.info(f"✓ Codec {codec_name} funcionando")
                        break
                    else:
                        test_writer.release()
                        self.logger.warning(f"✗ Codec {codec_name} falhou")
                        
                except Exception as e:
                    self.logger.warning(f"✗ Erro com codec {codec_name}: {e}")
                    continue
            
            if self.video_writer and self.video_writer.isOpened():
                self.recording = True
                self.logger.info(f"✓ Gravação iniciada: {output_file}")
                self.logger.info(f"✓ Codec utilizado: {successful_codec}")
                self.logger.info(f"✓ Resolução: {width}x{height} @ {rec_fps} FPS")
            else:
                self.logger.error("❌ Falha ao inicializar gravação com todos os codecs")
                self.logger.error("Verifique se o OpenCV foi compilado com suporte a codecs de vídeo")
                self.video_writer = None
                
        except Exception as e:
            self.logger.error(f"Erro ao iniciar gravação: {e}")
            self.video_writer = None
    
    def _stop_recording(self):
        """Para gravação de vídeo."""
        if self.video_writer:
            try:
                self.video_writer.release()
                self.logger.info("✓ Gravação finalizada com sucesso")
            except Exception as e:
                self.logger.error(f"Erro ao finalizar gravação: {e}")
            finally:
                self.video_writer = None
                self.recording = False
        else:
            self.logger.info("Nenhuma gravação ativa para parar")
    
    def _inference_loop(self):
        """Loop de inferência em thread separada."""
        self.logger.info("Thread de inferência iniciada")
        
        while self.running:
            try:
                # Obter frame da câmera
                frame = self.camera.get_frame(timeout=0.1)
                
                if frame is None:
                    continue
                
                # Verificar se está pausado
                if self.ui and self.ui.is_paused():
                    time.sleep(0.1)
                    continue
                
                # Processar frame
                annotated_frame, stats = self.detector.process_frame(frame)
                
                # Calcular FPS
                self.fps_counter += 1
                elapsed = time.time() - self.fps_start_time
                if elapsed >= 1.0:
                    self.current_fps = self.fps_counter / elapsed
                    self.fps_counter = 0
                    self.fps_start_time = time.time()
                
                # Atualizar stats com FPS
                stats["fps"] = self.current_fps
                stats["inference_ms"] = stats.get("inference_time_ms", 0)
                stats["capture_fps"] = self.camera.capture_fps if self.camera else 0.0
                
                # Atualizar UI
                if self.ui:
                    self.ui.update_frame(annotated_frame)
                    self.ui.update_stats(stats)
                
                # Gravar se habilitado
                if self.recording and self.video_writer:
                    try:
                        # Verificar se o VideoWriter ainda está válido
                        if self.video_writer.isOpened():
                            self.video_writer.write(annotated_frame)
                        else:
                            self.logger.warning("VideoWriter não está mais aberto, parando gravação")
                            self._stop_recording()
                    except Exception as e:
                        self.logger.error(f"Erro ao gravar frame: {e}")
                        self._stop_recording()
                
            except Exception as e:
                self.logger.error(f"Erro no loop de inferência: {e}", exc_info=True)
                time.sleep(0.1)
        
        self.logger.info("Thread de inferência finalizada")
    
    def start(self):
        """Inicia captura e inferência."""
        if self.running:
            return
        
        self.running = True
        
        # Iniciar captura da câmera
        if self.camera:
            self.camera.start_capture()
        
        # Iniciar thread de inferência
        self.inference_thread = threading.Thread(target=self._inference_loop, daemon=True)
        self.inference_thread.start()
        
        # Reset FPS counter
        self.fps_counter = 0
        self.fps_start_time = time.time()
        
        self.logger.info("Sistema iniciado")
    
    def stop(self):
        """Para captura e inferência."""
        if not self.running:
            return
        
        self.running = False
        
        # Parar gravação se ativa
        if self.recording:
            self._stop_recording()
        
        # Aguardar threads
        if self.inference_thread:
            self.inference_thread.join(timeout=2.0)
        
        # Parar câmera
        if self.camera:
            self.camera.stop_capture()
        
        # Salvar parâmetros antes de parar
        if self.detector:
            self.detector.save_parameters()
            
            # Gerar e exibir sumário final de estatísticas
            self._display_final_statistics()
        
        self.logger.info("Sistema parado")
    
    def run(self):
        """Executa a aplicação."""
        try:
            # Inicializar UI PRIMEIRO para que apareça mesmo se houver erros
            self._init_ui()
            
            # FORÇAR a janela a aparecer ANTES de qualquer outra coisa
            self.ui.root.update_idletasks()
            self.ui.root.deiconify()
            self.ui.root.lift()
            self.ui.root.focus_force()
            self.ui.root.update()
            
            # Pequeno delay para garantir que a janela apareça
            import time
            time.sleep(0.1)
            self.ui.root.update()
            
            # Mostrar mensagens de status na UI
            self.ui.set_status("Inicializando câmera...", "blue")
            self.ui.root.update()  # Atualizar UI para mostrar mensagem
            
            # Inicializar componentes
            camera_ok = False
            detector_ok = False
            
            if not self._init_camera():
                self.logger.error("Falha ao inicializar câmera. Verifique a conexão.")
                self.ui.set_status("ERRO: Câmera não encontrada! Verifique a conexão.", "red")
            else:
                camera_ok = True
                self.ui.set_status("Câmera inicializada com sucesso", "green")
            
            self.ui.root.update()  # Atualizar UI
            
            if camera_ok:
                self.ui.set_status("Inicializando detector...", "blue")
                self.ui.root.update()
                
                if not self._init_detector():
                    self.logger.error("Falha ao inicializar detector. Verifique os modelos.")
                    self.ui.set_status("ERRO: Detector não pode ser inicializado! Verifique os modelos.", "red")
                else:
                    detector_ok = True
                    self.ui.set_status("Sistema pronto! Clique em 'Iniciar' para começar.", "green")
            
            # Log de benchmark inicial (apenas se tudo OK)
            if camera_ok and detector_ok:
                self._log_benchmark()
            
            # Executar UI (bloqueia até fechar)
            self.logger.info("Iniciando loop principal da UI...")
            self.ui.run()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Erro na execução: {e}", exc_info=True)
            # Tentar mostrar erro na UI se ela foi criada
            if self.ui:
                try:
                    self.ui.set_status(f"ERRO: {str(e)}", "red")
                    self.ui.root.update()
                    self.ui.run()  # Mostrar UI mesmo com erro
                except:
                    pass
            return False
        
        finally:
            self.cleanup()
    
    def _test_recording_capability(self):
        """Testa a capacidade de gravação do sistema."""
        self.logger.info("Testando capacidade de gravação...")
        
        try:
            # Criar diretório de teste
            test_dir = Path("recordings")
            test_dir.mkdir(exist_ok=True)
            
            # Arquivo de teste
            test_file = test_dir / "test_recording.mp4"
            
            # Dimensões de teste
            width, height = 640, 480
            fps = 30
            
            # Testar codecs disponíveis (priorizando MP4 funcionando)
            codecs_to_test = [
                ("mp4v", "mp4v"),  # MPEG-4 - padrão MP4 funcionando
                ("MJPG", "MJPG"),  # Motion JPEG - fallback
                ("H264", "H264"),  # H.264 - pode não estar disponível
                ("XVID", "XVID"),  # XVID - alternativa
            ]
            
            working_codecs = []
            
            for codec_name, fourcc_str in codecs_to_test:
                try:
                    fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
                    test_writer = cv2.VideoWriter(
                        str(test_file),
                        fourcc,
                        fps,
                        (width, height)
                    )
                    
                    if test_writer.isOpened():
                        # Criar frame de teste
                        test_frame = np.zeros((height, width, 3), dtype=np.uint8)
                        test_writer.write(test_frame)
                        test_writer.release()
                        
                        # Verificar se arquivo foi criado
                        if test_file.exists() and test_file.stat().st_size > 0:
                            working_codecs.append(codec_name)
                            test_file.unlink()  # Remover arquivo de teste
                            self.logger.info(f"✓ Codec {codec_name}: FUNCIONANDO")
                        else:
                            self.logger.warning(f"✗ Codec {codec_name}: Arquivo não criado")
                    else:
                        self.logger.warning(f"✗ Codec {codec_name}: Não suportado")
                        
                except Exception as e:
                    self.logger.warning(f"✗ Codec {codec_name}: Erro - {e}")
                    continue
            
            if working_codecs:
                self.logger.info(f"✓ Codecs funcionais: {', '.join(working_codecs)}")
                return True
            else:
                self.logger.error("❌ Nenhum codec de vídeo funcionando!")
                self.logger.error("Instale codecs de vídeo ou recompile OpenCV com suporte a codecs")
                return False
                
        except Exception as e:
            self.logger.error(f"Erro no teste de gravação: {e}")
            return False
    
    def _log_benchmark(self):
        """Registra informações de benchmark."""
        self.logger.info("="*60)
        self.logger.info("BENCHMARK INICIAL")
        self.logger.info("="*60)
        
        if self.camera:
            info = self.camera.get_info()
            self.logger.info(f"Câmera: {info['width']}x{info['height']} @ {info['fps']} FPS")
            self.logger.info(f"PixelFormat: {info['pixel_format']}")
        
        if self.detector:
            perf = self.detector.get_performance_stats()
            self.logger.info(f"Device: {perf['device']}")
            self.logger.info(f"Imgsz: {self.detector.imgsz}")
        
        # Testar gravação
        recording_ok = self._test_recording_capability()
        if not recording_ok:
            self.logger.warning("⚠️ Gravação pode não funcionar corretamente")
        
        self.logger.info("="*60)
    
    def _save_config(self, config_path: str = "config/app.yaml"):
        """Salva a configuração atual em arquivo YAML."""
        try:
            # Garantir que o diretório existe
            Path(config_path).parent.mkdir(exist_ok=True)
            
            # Salvar configuração
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
            
            self.logger.info(f"✓ Configuração salva em: {config_path}")
            
            # Log das mudanças principais
            cam_cfg = self.config.get("camera", {})
            self.logger.info(f"  - FPS: {cam_cfg.get('fps_target', 4)}")
            self.logger.info(f"  - Exposição: {cam_cfg.get('exposure_time', 5000)} µs")
            self.logger.info(f"  - Ganho: {cam_cfg.get('gain', 0)} dB")
            self.logger.info(f"  - Balance White Auto: {cam_cfg.get('balance_white_auto', 'Off')}")
            
            thresholds = self.config.get("thresholds", {})
            self.logger.info(f"  - Thresholds: smudge={thresholds.get('smudge_conf', 0.5):.2f}, "
                           f"simbolos={thresholds.get('simbolo_conf', 0.5):.2f}, "
                           f"blackdot={thresholds.get('blackdot_conf', 0.5):.2f}")
            
        except Exception as e:
            self.logger.error(f"Erro ao salvar configuração: {e}")
    
    def _display_final_statistics(self):
        """Exibe e exporta sumário final de estatísticas."""
        if not self.detector:
            return
        
        try:
            # Obter sumário
            summary = self.detector.get_final_statistics_summary()
            
            if summary["total_transfers_evaluated"] == 0:
                self.logger.info("Nenhum transfer foi avaliado durante esta execução.")
                return
            
            # Exibir sumário no log
            self.logger.info("="*70)
            self.logger.info("📊 SUMÁRIO FINAL DE ESTATÍSTICAS")
            self.logger.info("="*70)
            
            # Total de transfers avaliados
            total_transfers = summary["total_transfers_evaluated"]
            self.logger.info(f"\n🔄 Transfers Avaliados: {total_transfers}")
            
            # Estatísticas por classe
            self.logger.info("\n📦 Objetos Detectados por Classe:")
            transfers_by_class = summary["transfers_by_class"]
            
            for class_name, class_stats in transfers_by_class.items():
                count = class_stats.get("count", 0)
                total_objects = class_stats.get("total_objects", 0)
                
                if "ok" in class_stats and "no" in class_stats:
                    # Classes com OK/NO (FIFA, Simbolo, String)
                    ok_count = class_stats.get("ok", 0)
                    no_count = class_stats.get("no", 0)
                    self.logger.info(f"  {class_name.upper()}:")
                    self.logger.info(f"    - Transfers com detecção: {count}")
                    self.logger.info(f"    - Total objetos: {total_objects} (OK: {ok_count}, NO: {no_count})")
                else:
                    # Classes sem OK/NO (Blackdot, Smudge)
                    self.logger.info(f"  {class_name.upper()}:")
                    self.logger.info(f"    - Transfers com detecção: {count}")
                    self.logger.info(f"    - Total objetos: {total_objects}")
            
            # Erro mais frequente
            most_frequent = summary["most_frequent_error"]
            self.logger.info(f"\n❌ Erro Mais Frequente:")
            self.logger.info(f"  - Classe: {most_frequent['class']} ({most_frequent['type']})")
            self.logger.info(f"  - Quantidade: {most_frequent['count']} detecções")
            self.logger.info(f"  - Percentual: {most_frequent['percentage']:.1f}% do total de erros")
            
            # Detalhes por transfer
            objects_per_transfer = summary["objects_per_transfer"]
            if objects_per_transfer:
                self.logger.info(f"\n📋 Resumo por Transfer (primeiros 5):")
                for i, transfer_obj in enumerate(objects_per_transfer[:5]):
                    transfer_id = transfer_obj.get("transfer_id", 0)
                    self.logger.info(f"  Transfer #{transfer_id}:")
                    self.logger.info(f"    - Blackdot: {transfer_obj.get('blackdot', 0)}")
                    self.logger.info(f"    - Smudge: {transfer_obj.get('smudge', 0)}")
                    fifa = transfer_obj.get('fifa', {})
                    simbolo = transfer_obj.get('simbolo', {})
                    string = transfer_obj.get('string', {})
                    self.logger.info(f"    - FIFA: OK={fifa.get('ok', 0)}, NO={fifa.get('no', 0)}")
                    self.logger.info(f"    - Simbolo: OK={simbolo.get('ok', 0)}, NO={simbolo.get('no', 0)}")
                    self.logger.info(f"    - String: OK={string.get('ok', 0)}, NO={string.get('no', 0)}")
                
                if len(objects_per_transfer) > 5:
                    self.logger.info(f"    ... e mais {len(objects_per_transfer) - 5} transfers")
            
            self.logger.info("="*70)
            
            # Exportar para arquivo
            exported_file = self.detector.export_statistics_to_file()
            if exported_file:
                self.logger.info(f"✓ Estatísticas exportadas para: {exported_file}")
            else:
                self.logger.warning("⚠ Não foi possível exportar estatísticas para arquivo")
            
            # Atualizar UI com sumário
            if self.ui:
                try:
                    self.ui.update_statistics_summary(summary)
                    # Forçar atualização da janela de estatísticas se estiver aberta
                    if hasattr(self.ui, 'stats_window') and hasattr(self.ui.stats_window, 'winfo_exists'):
                        if self.ui.stats_window.winfo_exists():
                            self.ui._update_detailed_stats()
                except Exception as e:
                    self.logger.warning(f"Erro ao atualizar UI com sumário: {e}")
                
        except Exception as e:
            self.logger.error(f"Erro ao gerar sumário final: {e}", exc_info=True)
    
    def cleanup(self):
        """Limpa recursos e salva configurações."""
        self.logger.info("Limpando recursos...")
        
        self.stop()
        
        # Salvar configurações da UI
        if self.ui and hasattr(self, 'config_manager'):
            try:
                ui_settings = self.config_manager.get_ui_settings(self.ui)
                if ui_settings:
                    self.config_manager.save_settings(ui_settings)
                    self.logger.info("✓ Configurações da interface salvas")
            except Exception as e:
                self.logger.error(f"Erro ao salvar configurações da UI: {e}")
        
        # Salvar configurações atualizadas
        self._save_config()
        
        if self.camera:
            self.camera.close()
        
        # Gerar sumário final se ainda não foi gerado
        if self.detector:
            self._display_final_statistics()
        
        self.logger.info("Aplicação finalizada")


def main():
    """Função principal."""
    try:
        # Configurar variáveis de ambiente
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        
        print("="*70)
        print("Iniciando YOLO Detection System...")
        print("="*70)
        print()
        
        # Criar diretórios necessários
        Path("logs").mkdir(exist_ok=True)
        Path("recordings").mkdir(exist_ok=True)
        Path("models").mkdir(exist_ok=True)
        print("[OK] Diretorios criados")
        
        # Criar e executar aplicação
        print("[OK] Criando aplicacao...")
        app = YOLODetectionApp()
        print("[OK] Aplicacao criada, iniciando execucao...")
        print()
        
        success = app.run()
        
        if success:
            print()
            print("[OK] Aplicacao finalizada normalmente")
            sys.exit(0)
        else:
            print()
            print("[ERRO] Aplicacao finalizada com erros")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n[INFO] Interrompido pelo usuario")
        if 'app' in locals():
            app.cleanup()
        sys.exit(0)
    except Exception as e:
        print()
        print("="*70)
        print("[ERRO FATAL]")
        print("="*70)
        print(f"Erro: {e}")
        print()
        print("Traceback completo:")
        import traceback
        traceback.print_exc()
        print()
        print("="*70)
        input("Pressione ENTER para sair...")
        sys.exit(1)


if __name__ == "__main__":
    main()

