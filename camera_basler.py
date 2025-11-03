"""
Módulo de captura de câmera Basler via pypylon.
Suporta USB3 Vision com alta taxa de quadros.
"""

import logging
import threading
import queue
import time
from typing import Optional, Tuple
import numpy as np

try:
    from pypylon import pylon
    PYLON_AVAILABLE = True
except ImportError:
    PYLON_AVAILABLE = False
    logging.warning("pypylon não disponível. Instale com: pip install pypylon")


class BaslerCamera:
    """Gerenciador de câmera Basler USB3 Vision com pypylon."""
    
    def __init__(self, width: Optional[int] = None, height: Optional[int] = None, 
                 fps: int = 120, pixel_format: str = "Mono8",
                 exposure_time: int = 5000, gain: float = 0,
                 timeout_ms: int = 50, balance_white_auto: str = "Off"):
        """
        Inicializa a câmera Basler.
        
        Args:
            width: Largura do ROI (None = máximo)
            height: Altura do ROI (None = máximo)
            fps: Taxa de quadros alvo
            pixel_format: Formato de pixel (Mono8, BayerRG8, etc)
            exposure_time: Tempo de exposição em microsegundos
            gain: Ganho da câmera
            timeout_ms: Timeout para RetrieveResult
            balance_white_auto: Modo de balance white ("Off", "Once", "Continuous")
        """
        if not PYLON_AVAILABLE:
            raise RuntimeError("pypylon não está instalado")
        
        self.logger = logging.getLogger(__name__)
        self.camera: Optional[pylon.InstantCamera] = None
        self.converter: Optional[pylon.ImageFormatConverter] = None
        self.timeout_ms = timeout_ms
        self.running = False
        self.grab_thread: Optional[threading.Thread] = None
        self.frame_queue = queue.Queue(maxsize=8)
        
        self.width = width
        self.height = height
        self.fps = fps
        self.pixel_format = pixel_format
        self.exposure_time = exposure_time
        self.gain = gain
        self.balance_white_auto = balance_white_auto
        
        self.actual_width = 0
        self.actual_height = 0
        self.actual_fps = 0
        self.capture_fps = 0.0  # FPS de captura em tempo real
        
    def open(self) -> bool:
        """Abre e configura a câmera Basler."""
        try:
            # Criar instância da câmera
            tl_factory = pylon.TlFactory.GetInstance()
            devices = tl_factory.EnumerateDevices()
            
            if len(devices) == 0:
                self.logger.error("Nenhuma câmera Basler encontrada")
                return False
            
            self.logger.info(f"Câmera encontrada: {devices[0].GetFriendlyName()}")
            self.camera = pylon.InstantCamera(tl_factory.CreateFirstDevice())
            self.camera.Open()
            
            # Configurar formato de pixel
            try:
                self.camera.PixelFormat.SetValue(self.pixel_format)
                self.logger.info(f"PixelFormat definido: {self.pixel_format}")
            except Exception as e:
                self.logger.warning(f"Não foi possível definir PixelFormat: {e}")
            
            # CRÍTICO: Configurar modo de aquisição contínua
            try:
                self.camera.AcquisitionMode.SetValue("Continuous")
                self.logger.info(f"AcquisitionMode: Continuous")
            except Exception as e:
                self.logger.warning(f"Não foi possível definir AcquisitionMode: {e}")
            
            # Desativar automáticos para melhor performance
            try:
                if hasattr(self.camera, 'ExposureAuto'):
                    self.camera.ExposureAuto.SetValue("Off")
                if hasattr(self.camera, 'GainAuto'):
                    self.camera.GainAuto.SetValue("Off")
            except Exception as e:
                self.logger.warning(f"Erro ao desativar automáticos: {e}")
            
            # Configurar Balance White Auto
            try:
                if hasattr(self.camera, 'BalanceWhiteAuto'):
                    self.camera.BalanceWhiteAuto.SetValue(self.balance_white_auto)
                    self.logger.info(f"BalanceWhiteAuto: {self.balance_white_auto}")
            except Exception as e:
                self.logger.warning(f"Erro ao configurar BalanceWhiteAuto: {e}")
            
            # Definir exposição e ganho
            try:
                if hasattr(self.camera, 'ExposureTime'):
                    self.camera.ExposureTime.SetValue(self.exposure_time)
                    self.logger.info(f"ExposureTime: {self.exposure_time} µs")
            except Exception as e:
                self.logger.warning(f"Erro ao definir ExposureTime: {e}")
                
            try:
                if hasattr(self.camera, 'Gain'):
                    self.camera.Gain.SetValue(self.gain)
                    self.logger.info(f"Gain: {self.gain}")
            except Exception as e:
                self.logger.warning(f"Erro ao definir Gain: {e}")
            
            # Habilitar frame rate configurável
            try:
                if hasattr(self.camera, 'AcquisitionFrameRateEnable'):
                    self.camera.AcquisitionFrameRateEnable.SetValue(True)
                if hasattr(self.camera, 'AcquisitionFrameRate'):
                    max_fps = self.camera.AcquisitionFrameRate.Max
                    target_fps = min(self.fps, max_fps)
                    self.camera.AcquisitionFrameRate.SetValue(target_fps)
                    self.actual_fps = target_fps
                    self.logger.info(f"FPS configurado: {target_fps} (máx: {max_fps})")
            except Exception as e:
                self.logger.warning(f"Erro ao configurar FPS: {e}")
                self.actual_fps = self.fps
            
            # Configurar ROI se especificado
            # IMPORTANTE: Seguindo o padrão do código de referência que funciona,
            # apenas configuramos Width e Height SEM tocar em OffsetX/OffsetY
            # Isso evita deslocamento na captura
            if self.width and self.height:
                try:
                    max_width = self.camera.Width.Max
                    max_height = self.camera.Height.Max
                    
                    target_width = min(self.width, max_width)
                    target_height = min(self.height, max_height)
                    
                    # Ajustar para incrementos válidos
                    if self.camera.Width.Inc > 1:
                        target_width = (target_width // self.camera.Width.Inc) * self.camera.Width.Inc
                    if self.camera.Height.Inc > 1:
                        target_height = (target_height // self.camera.Height.Inc) * self.camera.Height.Inc
                    
                    # CRÍTICO: Apenas definir Width e Height, SEM configurar offsets
                    # O código de referência mostra que isso é suficiente e evita deslocamento
                    self.camera.Width.SetValue(target_width)
                    self.camera.Height.SetValue(target_height)
                    
                    # Verificar valores aplicados
                    actual_width = self.camera.Width.GetValue()
                    actual_height = self.camera.Height.GetValue()
                    
                    self.logger.info(f"✓ Resolução configurada: {actual_width}x{actual_height}")
                    
                    if actual_width != target_width or actual_height != target_height:
                        self.logger.warning(f"⚠ Dimensões aplicadas ({actual_width}x{actual_height}) diferem das desejadas ({target_width}x{target_height})")
                    
                except Exception as e:
                    self.logger.warning(f"Erro ao configurar resolução: {e}")
            
            # Obter dimensões reais
            self.actual_width = self.camera.Width.GetValue()
            self.actual_height = self.camera.Height.GetValue()
            self.logger.info(f"Resolução final: {self.actual_width}x{self.actual_height}")
            
            # Configurar número de buffers
            try:
                self.camera.MaxNumBuffer = 5
                self.logger.info(f"MaxNumBuffer: 5")
            except Exception as e:
                self.logger.warning(f"Não foi possível definir MaxNumBuffer: {e}")
            
            # Configurar conversor de imagem (igual ao código de referência que funciona)
            self.converter = pylon.ImageFormatConverter()
            self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed
            # CRÍTICO: Usar LsbAligned como no código de referência que funciona (evita deslocamento)
            self.converter.OutputBitAlignment = pylon.OutputBitAlignment_LsbAligned
            
            # CRÍTICO: Parar qualquer grabbing existente antes de iniciar
            # Isso garante que as configurações de ROI sejam aplicadas corretamente
            try:
                if self.camera.IsGrabbing():
                    self.camera.StopGrabbing()
                    time.sleep(0.05)  # Aguardar para garantir que pare completamente
            except:
                pass  # Se não estiver grabbing, continuar
            
            # Iniciar aquisição com estratégia LatestImageOnly
            # Após configurar ROI, o frame já vem com offset correto aplicado pela câmera
            self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
            
            # Aguardar um frame para garantir que a configuração está ativa
            time.sleep(0.1)
            
            self.logger.info("Câmera Basler aberta e configurada com sucesso")
            return True
            
        except Exception as e:
            self.logger.error(f"Erro ao abrir câmera Basler: {e}")
            if self.camera and self.camera.IsOpen():
                self.camera.Close()
            return False
    
    def grab_frame(self) -> Optional[np.ndarray]:
        """Captura um frame da câmera SEM nenhuma transformação adicional."""
        if not self.camera or not self.camera.IsGrabbing():
            return None
        
        try:
            grab_result = self.camera.RetrieveResult(
                self.timeout_ms, 
                pylon.TimeoutHandling_Return
            )
            
            if grab_result and grab_result.GrabSucceeded():
                # IMPORTANTE: Verificar dimensões do frame capturado vs configuração da câmera
                # Se houver discrepância, pode indicar problema de offset
                captured_width = grab_result.Width
                captured_height = grab_result.Height
                
                # Log de diagnóstico apenas a cada 300 frames para não poluir (aprox. 1x por minuto a 4fps)
                if hasattr(self, '_debug_frame_count'):
                    self._debug_frame_count += 1
                else:
                    self._debug_frame_count = 1
                
                if self._debug_frame_count % 300 == 1:
                    actual_width = self.camera.Width.GetValue() if self.camera else 0
                    actual_height = self.camera.Height.GetValue() if self.camera else 0
                    actual_offset_x = self.camera.OffsetX.GetValue() if self.camera else 0
                    actual_offset_y = self.camera.OffsetY.GetValue() if self.camera else 0
                    
                    if captured_width != actual_width or captured_height != actual_height:
                        self.logger.warning(f"⚠ DISCREPÂNCIA: Frame capturado {captured_width}x{captured_height} vs config {actual_width}x{actual_height}")
                        self.logger.warning(f"   Offset configurado: ({actual_offset_x}, {actual_offset_y})")
                
                # Converter para BGR sem aplicar transformações (frame já vem com offset correto da câmera)
                # O ImageFormatConverter apenas converte o formato de pixel, não altera posicionamento
                image = self.converter.Convert(grab_result)
                img_array = image.GetArray()
                
                # IMPORTANTE: O frame já vem com o ROI e offset aplicados pela câmera
                # Não aplicar nenhuma transformação adicional (crop, resize, etc)
                # O array retornado deve ter exatamente as dimensões configuradas na câmera
                grab_result.Release()
                return img_array
            
            if grab_result:
                grab_result.Release()
                
        except Exception as e:
            # Log apenas erros não esperados
            if "timeout" not in str(e).lower():
                self.logger.debug(f"Erro ao capturar frame: {e}")
        
        return None
    
    def _grab_loop(self):
        """Loop de captura em thread separada."""
        self.logger.info("Thread de captura iniciada")
        frame_count = 0
        error_count = 0
        last_log = time.time()
        last_recovery = time.time()
        fps_start = time.time()
        fps_frame_count = 0
        
        while self.running:
            try:
                frame = self.grab_frame()
                
                if frame is not None:
                    frame_count += 1
                    fps_frame_count += 1
                    error_count = 0
                    
                    # Calcular FPS de captura
                    elapsed = time.time() - fps_start
                    if elapsed >= 1.0:
                        self.capture_fps = fps_frame_count / elapsed
                        fps_frame_count = 0
                        fps_start = time.time()
                    
                    # Adicionar à fila sem bloquear
                    try:
                        self.frame_queue.put_nowait(frame)
                    except queue.Full:
                        # Descartar frame se fila cheia
                        pass
                else:
                    error_count += 1
                    # Recovery apenas se muitos erros E já passou tempo suficiente desde último recovery
                    if error_count > 500 and (time.time() - last_recovery) > 5.0:
                        self.logger.warning(f"Muitos erros de captura ({error_count}), tentando recovery...")
                        self._recovery()
                        error_count = 0
                        last_recovery = time.time()
                
                # Log periódico de sucesso
                if time.time() - last_log > 10.0:
                    if frame_count > 0:
                        self.logger.info(f"Frames capturados: {frame_count}, FPS: {self.capture_fps:.1f}, fila: {self.frame_queue.qsize()}")
                    else:
                        self.logger.warning(f"Nenhum frame capturado nos últimos 10s (erros: {error_count})")
                    last_log = time.time()
                    
            except Exception as e:
                self.logger.error(f"Erro no loop de captura: {e}")
                time.sleep(0.1)
        
        self.logger.info("Thread de captura finalizada")
    
    def _recovery(self):
        """Tenta recuperar a câmera em caso de erros."""
        try:
            if self.camera and self.camera.IsGrabbing():
                self.camera.StopGrabbing()
                time.sleep(0.2)
                self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
                self.logger.info("Recovery da câmera realizado")
        except Exception as e:
            self.logger.error(f"Erro no recovery: {e}")
    
    def start_capture(self):
        """Inicia captura em thread separada."""
        if self.running:
            return
        
        self.running = True
        self.grab_thread = threading.Thread(target=self._grab_loop, daemon=True)
        self.grab_thread.start()
        self.logger.info("Captura iniciada")
    
    def get_frame(self, timeout: float = 0.1) -> Optional[np.ndarray]:
        """Obtém próximo frame da fila."""
        try:
            return self.frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def stop_capture(self):
        """Para a captura."""
        self.running = False
        if self.grab_thread:
            self.grab_thread.join(timeout=2.0)
    
    def close(self):
        """Fecha a câmera."""
        self.stop_capture()
        
        if self.camera:
            try:
                if self.camera.IsGrabbing():
                    self.camera.StopGrabbing()
                self.camera.Close()
                self.logger.info("Câmera Basler fechada")
            except Exception as e:
                self.logger.error(f"Erro ao fechar câmera: {e}")
        
        self.camera = None
        self.converter = None
    
    def set_balance_white_auto(self, mode: str) -> bool:
        """
        Define o modo de Balance White Auto.
        
        Args:
            mode: "Off", "Once", ou "Continuous"
        
        Returns:
            True se sucesso, False caso contrário
        """
        if not self.camera or not hasattr(self.camera, 'BalanceWhiteAuto'):
            self.logger.warning("BalanceWhiteAuto não disponível nesta câmera")
            return False
        
        try:
            self.camera.BalanceWhiteAuto.SetValue(mode)
            self.balance_white_auto = mode
            self.logger.info(f"BalanceWhiteAuto atualizado: {mode}")
            return True
        except Exception as e:
            self.logger.error(f"Erro ao atualizar BalanceWhiteAuto: {e}")
            return False
    
    def get_info(self) -> dict:
        """Retorna informações da câmera."""
        return {
            "width": self.actual_width,
            "height": self.actual_height,
            "fps": self.actual_fps,
            "pixel_format": self.pixel_format,
            "balance_white_auto": self.balance_white_auto,
            "is_grabbing": self.camera.IsGrabbing() if self.camera else False,
            "queue_size": self.frame_queue.qsize()
        }

