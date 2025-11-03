"""
Módulo de inferência YOLO com segmentação ROI e detecção multi-modelo.
Implementa lógica de contagem por "transfer" com tracking de IOUs.
"""

import logging
import time
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import cv2
import torch

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    logging.warning("Ultralytics não disponível. Instale com: pip install ultralytics")


class YOLODetector:
    """Sistema de detecção YOLO multi-modelo com ROI."""
    
    def __init__(self, config: dict, device: str = "cuda:0"):
        """
        Inicializa o sistema de detecção.
        
        Args:
            config: Dicionário de configuração
            device: Dispositivo PyTorch (cuda:0, cpu)
        """
        if not ULTRALYTICS_AVAILABLE:
            raise RuntimeError("Ultralytics não está instalado")
        
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.device = device
        
        # Verificar disponibilidade CUDA
        if "cuda" in device:
            if not torch.cuda.is_available():
                self.logger.warning("CUDA não disponível, usando CPU")
                self.device = "cpu"
            else:
                # Otimizações específicas para RTX 3050
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                
                # Configurar memória para RTX 3050 (4GB VRAM)
                if torch.cuda.get_device_properties(0).total_memory < 6 * 1024**3:  # < 6GB
                    torch.cuda.empty_cache()
                    torch.cuda.set_per_process_memory_fraction(0.8)  # Usar 80% da VRAM
                
                self.logger.info(f"CUDA disponível: {torch.cuda.get_device_name(0)}")
                self.logger.info(f"CUDA version: {torch.version.cuda}")
                self.logger.info(f"VRAM total: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
        
        # Modelos
        self.seg_model = None
        self.smudge_model = None
        self.simbolos_model = None
        self.blackdot_model = None
        
        # Controles de ativação dos modelos
        self.model_enabled = {
            "seg": True,
            "smudge": True,
            "simbolos": True,
            "blackdot": True
        }
        
        # Parâmetros de inferência
        self.imgsz = config.get("inference", {}).get("imgsz", 640)
        self.max_det = config.get("inference", {}).get("max_det", 100)
        
        # Thresholds - CORRIGIDOS para reduzir conflitos
        self.roi_conf = config.get("roi", {}).get("conf", 0.5)  # Default: 0.5 (50%)
        self.roi_iou = config.get("roi", {}).get("iou", 0.45)
        self.roi_min_pixels = config.get("roi", {}).get("min_pixels", 2000)
        
        thresholds = config.get("thresholds", {})
        # Todos os thresholds com default de 0.5 (50%)
        self.smudge_conf = thresholds.get("smudge_conf", 0.5)  # Default: 0.5 (50%)
        self.smudge_iou = thresholds.get("smudge_iou", 0.45)
        self.simbolo_conf = thresholds.get("simbolo_conf", 0.5)  # Default: 0.5 (50%)
        self.simbolo_iou = thresholds.get("simbolo_iou", 0.45)
        self.blackdot_conf = thresholds.get("blackdot_conf", 0.5)  # Default: 0.5 (50%)
        self.blackdot_iou = thresholds.get("blackdot_iou", 0.45)
        
        # Transfer tracking baseado em ROI
        transfer_cfg = config.get("transfer", {})
        self.ABSENT_TO_NEW = transfer_cfg.get("absent_to_new", 8)
        self.IOU_NEW_THRESH = transfer_cfg.get("iou_new_thresh", 0.30)
        
        # Controle de transfer baseado em ROI
        self.transfer_count = 0
        self.frames_without_roi = 0
        self.last_roi_bbox = None
        self.current_transfer_active = False  # Se há um transfer ativo
        
        # Sistema de estabilização de detecções
        self.detection_history = {
            "smudge": [],
            "simbolos": [],
            "blackdot": []
        }
        self.stabilization_window = 8  # Frames para estabilização (aumentado para mais estabilidade)
        self.min_detection_confidence = 0.5  # Confiança mínima aumentada para reduzir falsos positivos
        
        # Sistema de filtros de sobreposição - OTIMIZADO para reduzir conflitos
        self.overlap_threshold = 0.3  # IOU reduzido para ser mais rigoroso
        # Prioridade de classes: blackdot (0) > simbolos (1) > smudge (2)
        # - blackdot tem prioridade sobre smudge ✓
        # - simbolos (incluindo FIFA, Simbolo, String do modelo) tem prioridade sobre smudge ✓
        # - smudge (FIFA do modelo smudge) tem menor prioridade ✓
        self.class_priority = ["blackdot", "simbolos", "smudge"]
        
        # Cache de parâmetros para persistência
        self.parameter_cache = {
            "thresholds": {},
            "model_enabled": {},
            "inference_params": {}
        }
        self.current_transfer_start_frame = 0  # Frame de início do transfer
        self.current_transfer_frames = 0  # Número de frames do transfer atual
        
        # Estatísticas aprimoradas de transfer
        self.transfer_stats = {
            "total_evaluated": 0,      # Total de transfers avaliados
            "total_approved": 0,        # Total de transfers aprovados
            "total_rejected": 0,        # Total de transfers reprovados
            "current_transfer": None,   # Transfer atual em andamento
            "transfer_history": []      # Histórico de transfers
        }
        
        # Estabilização de bounding boxes
        self.bbox_history = []          # Histórico de bboxes para suavização
        self.bbox_smoothing_frames = 5  # Número de frames para suavização
        self.bbox_confidence_threshold = 0.3  # Threshold mínimo para considerar bbox válida
        
        # Estatísticas por transfer
        self.current_transfer_stats = {
            "smudge": [],  # Lista de contagens por frame
            "simbolos": [],  # Lista de contagens por frame
            "blackdot": [],  # Lista de contagens por frame
            "fifa_ok": [],  # FIFA OK por frame
            "fifa_no": [],  # FIFA NO por frame
            "simbolo_ok": [],  # Simbolo OK por frame
            "simbolo_no": [],  # Simbolo NO por frame
            "string_ok": [],  # String OK por frame
            "string_no": []  # String NO por frame
        }
        
        self.transfer_averages = {
            "smudge": 0.0,
            "simbolos": 0.0,
            "blackdot": 0.0
        }
        
        # Sistema de média móvel para estabilizar classe predominante
        self.class_history = []  # Histórico das últimas 10 frames
        self.moving_average_window = 10  # Janela de média móvel
        self.predominant_class = "Nenhuma"  # Classe predominante atual
        self.predominant_class_confidence = 0.0  # Confiança da classe predominante
        
        # Mapeamento de classes corretas
        self.class_mapping = {
            "smudge": "Smudge",
            "simbolos": "Símbolo", 
            "blackdot": "BlackDot"
        }
        
        # Estabilização de detecções de smudge
        self.smudge_history = []  # Histórico de detecções
        self.smudge_stability_threshold = 5  # Aumentado para mais estabilidade
        self.smudge_confidence_buffer = []  # Buffer de confianças
        self.smudge_stable_detection = None  # Detecção estável atual
        
        # Estabilização de detecções de símbolos
        self.symbols_history = []  # Histórico de detecções de símbolos
        self.symbols_stability_threshold = 5  # Aumentado para mais estabilidade
        self.symbols_confidence_buffer = []  # Buffer de confianças
        self.symbols_stable_detection = None  # Detecção estável atual
        
        # Estabilização de classes específicas
        self.fifa_history = []  # Histórico de detecções FIFA
        self.string_history = []  # Histórico de detecções String
        self.fifa_stable_detection = None  # Detecção estável FIFA
        self.string_stable_detection = None  # Detecção estável String
        
        # Performance tracking
        self.last_inference_time = 0.0
        self.avg_inference_time = 0.0
        self.frame_count = 0
        
    def load_models(self) -> bool:
        """Carrega todos os modelos YOLO."""
        try:
            models_cfg = self.config.get("models", {})
            
            self.logger.info("="*60)
            self.logger.info("CARREGANDO MODELOS YOLO")
            self.logger.info("="*60)
            
            # Modelo de segmentação (ROI) - Crop_Fifa_best.pt
            seg_path = models_cfg.get("seg", "models/Crop_Fifa_best.pt")
            self.logger.info(f"[1/4] Carregando modelo de SEGMENTAÇÃO ROI: {seg_path}")
            self.seg_model = YOLO(seg_path)
            self.seg_model.to(self.device)
            
            # Obter informações do modelo de segmentação
            if hasattr(self.seg_model, 'names'):
                self.logger.info(f"      ✓ Classes do modelo ROI: {self.seg_model.names}")
            self.logger.info(f"      ✓ Modelo ROI carregado com SUCESSO")
            
            # Modelo de smudge
            smudge_path = models_cfg.get("smudge", "models/smudge.pt")
            self.logger.info(f"[2/4] Carregando modelo de SMUDGE: {smudge_path}")
            self.smudge_model = YOLO(smudge_path)
            self.smudge_model.to(self.device)
            
            if hasattr(self.smudge_model, 'names'):
                self.logger.info(f"      ✓ Classes do modelo Smudge: {self.smudge_model.names}")
            self.logger.info(f"      ✓ Modelo Smudge carregado com SUCESSO")
            
            # Modelo de símbolos
            simbolos_path = models_cfg.get("simbolos", "models/simbolos.pt")
            self.logger.info(f"[3/4] Carregando modelo de SÍMBOLOS: {simbolos_path}")
            self.simbolos_model = YOLO(simbolos_path)
            self.simbolos_model.to(self.device)
            
            if hasattr(self.simbolos_model, 'names'):
                self.logger.info(f"      ✓ Classes do modelo Símbolos: {self.simbolos_model.names}")
            self.logger.info(f"      ✓ Modelo Símbolos carregado com SUCESSO")
            
            # Modelo de blackdot
            blackdot_path = models_cfg.get("blackdot", "models/blackdot.pt")
            self.logger.info(f"[4/4] Carregando modelo de BLACKDOT: {blackdot_path}")
            self.blackdot_model = YOLO(blackdot_path)
            self.blackdot_model.to(self.device)
            
            if hasattr(self.blackdot_model, 'names'):
                self.logger.info(f"      ✓ Classes do modelo BlackDot: {self.blackdot_model.names}")
            self.logger.info(f"      ✓ Modelo BlackDot carregado com SUCESSO")
            
            self.logger.info("="*60)
            self.logger.info("✓ TODOS OS 4 MODELOS CARREGADOS COM SUCESSO")
            self.logger.info(f"✓ Device: {self.device}")
            self.logger.info(f"✓ ImgSz: {self.imgsz}")
            self.logger.info("="*60)
            
            return True
            
        except Exception as e:
            self.logger.error(f"✗ ERRO ao carregar modelos: {e}", exc_info=True)
            return False
    
    def warmup(self):
        """Aquece os modelos com inferência dummy."""
        self.logger.info("Aquecendo modelos...")
        try:
            dummy_input = torch.zeros(1, 3, self.imgsz, self.imgsz, device=self.device)
            
            # Warm-up com imagem dummy
            dummy_np = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
            
            if self.seg_model:
                _ = self.seg_model.predict(dummy_np, imgsz=self.imgsz, verbose=False)
            if self.smudge_model:
                _ = self.smudge_model.predict(dummy_np, imgsz=self.imgsz, verbose=False)
            if self.simbolos_model:
                _ = self.simbolos_model.predict(dummy_np, imgsz=self.imgsz, verbose=False)
            if self.blackdot_model:
                _ = self.blackdot_model.predict(dummy_np, imgsz=self.imgsz, verbose=False)
            
            self.logger.info("Warm-up concluído")
            
        except Exception as e:
            self.logger.warning(f"Erro no warm-up: {e}")
    
    def compute_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """Calcula IOU entre duas bounding boxes."""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
            return 0.0
        
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def bbox_iou(self, bbox1: Optional[Tuple[int, int, int, int]], bbox2: Optional[Tuple[int, int, int, int]]) -> float:
        """
        Calcula IOU entre dois bounding boxes (baseado no código MacBook).
        
        Args:
            bbox1: (x, y, w, h) ou None
            bbox2: (x, y, w, h) ou None
        
        Returns:
            IOU entre 0.0 e 1.0
        """
        if bbox1 is None or bbox2 is None:
            return 0.0
        
        x0, y0, w0, h0 = bbox1
        x1, y1, w1, h1 = bbox2
        
        a0 = w0 * h0
        a1 = w1 * h1
        xa0, ya0, xa1, ya1 = x0, y0, x0 + w0, y0 + h0
        xb0, yb0, xb1, yb1 = x1, y1, x1 + w1, y1 + h1
        xi0, yi0 = max(xa0, xb0), max(ya0, yb0)
        xi1, yi1 = min(xa1, xb1), min(ya1, yb1)
        inter = max(0, xi1 - xi0) * max(0, yi1 - yi0)
        union = a0 + a1 - inter + 1e-9
        return inter / union

    def box_center_inside(self, bbox, box_xyxy):
        """Verifica se o centro de um box está dentro de um bbox (baseado no MacBook)."""
        if bbox is None or box_xyxy is None:
            return False
        x0, y0, bw, bh = bbox
        rx0, ry0, rx1, ry1 = x0, y0, x0 + bw, y0 + bh
        x1, y1, x2, y2 = box_xyxy
        cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
        return (rx0 <= cx <= rx1) and (ry0 <= cy <= ry1)
    
    def is_detection_inside_roi(self, detection_bbox: Tuple[int, int, int, int], roi_mask: Optional[np.ndarray], roi_bbox: Optional[Tuple[int, int, int, int]]) -> bool:
        """
        Verifica se uma detecção está dentro da ROI usando máscara ou bbox.
        OTIMIZADO para performance: verificações rápidas primeiro, cálculos custosos apenas se necessário.
        
        Args:
            detection_bbox: Bounding box da detecção (x1, y1, x2, y2)
            roi_mask: Máscara da ROI no frame completo
            roi_bbox: Bounding box da ROI (x, y, w, h)
            
        Returns:
            True se a detecção está dentro da ROI
        """
        if roi_bbox is None:
            return False
        
        x1, y1, x2, y2 = detection_bbox
        roi_x, roi_y, roi_w, roi_h = roi_bbox
        
        # Calcula centro uma vez
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        
        # VERIFICAÇÃO RÁPIDA 1: Verificar se o centro está dentro do bbox da ROI
        if not (roi_x <= center_x <= roi_x + roi_w and roi_y <= center_y <= roi_y + roi_h):
            return False
        
        # Se temos máscara, usar para validação mais precisa
        if roi_mask is not None:
            # VERIFICAÇÃO RÁPIDA 2: Verificar se o centro está na máscara (muito mais rápido que calcular área)
            if 0 <= center_y < roi_mask.shape[0] and 0 <= center_x < roi_mask.shape[1]:
                if roi_mask[center_y, center_x] > 0:
                    return True
            
            # VERIFICAÇÃO CUSTOSA: Só calcular área de interseção se o centro não estiver na máscara
            # Mas ainda assim, otimizar verificando se há interseção válida primeiro
            x_min = max(x1, 0)
            y_min = max(y1, 0)
            x_max = min(x2, roi_mask.shape[1])
            y_max = min(y2, roi_mask.shape[0])
            
            if x_max > x_min and y_max > y_min:
                detection_area = (x2 - x1) * (y2 - y1)
                if detection_area > 0:
                    # Calcular área de interseção (operação custosa, mas apenas se necessário)
                    # Otimização: usar apenas a região de interseção
                    mask_region = roi_mask[y_min:y_max, x_min:x_max]
                    intersection_area = np.count_nonzero(mask_region)  # Mais rápido que np.sum(... > 0)
                    
                    # Se pelo menos 50% da detecção está dentro da máscara
                    if (intersection_area / detection_area) >= 0.5:
                        return True
            
            # Se há máscara mas não passou em nenhuma verificação, retornar False
            return False
        
        # Se não há máscara, usar apenas verificação de bbox (já verificamos que o centro está dentro)
        return True

    def is_symbol_ok(self, label: str):
        """Verifica se um símbolo é OK (baseado no MacBook)."""
        return 'ok' in label.lower()
    
    def _smooth_bbox(self, current_bbox: Tuple[int, int, int, int], confidence: float) -> Tuple[int, int, int, int]:
        """Suaviza bounding box usando histórico para estabilidade."""
        if confidence < self.bbox_confidence_threshold:
            # Se confiança baixa, usar bbox anterior se disponível
            if self.bbox_history:
                return self.bbox_history[-1]
            return current_bbox
        
        # Adicionar bbox atual ao histórico
        self.bbox_history.append(current_bbox)
        
        # Manter apenas os últimos N frames
        if len(self.bbox_history) > self.bbox_smoothing_frames:
            self.bbox_history.pop(0)
        
        # Se não temos histórico suficiente, retornar atual
        if len(self.bbox_history) < 2:
            return current_bbox
        
        # Calcular média ponderada dos últimos frames
        x1_sum = y1_sum = x2_sum = y2_sum = 0
        total_weight = 0
        
        for i, bbox in enumerate(self.bbox_history):
            weight = (i + 1) / len(self.bbox_history)  # Peso maior para frames mais recentes
            x1, y1, x2, y2 = bbox
            
            x1_sum += x1 * weight
            y1_sum += y1 * weight
            x2_sum += x2 * weight
            y2_sum += y2 * weight
            total_weight += weight
        
        # Retornar bbox suavizada
        smoothed_bbox = (
            int(x1_sum / total_weight),
            int(y1_sum / total_weight),
            int(x2_sum / total_weight),
            int(y2_sum / total_weight)
        )
        
        return smoothed_bbox
    
    def _calculate_iou(self, box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
        """
        Calcula o IOU (Intersection over Union) entre duas bounding boxes.
        
        Args:
            box1: (x1, y1, x2, y2)
            box2: (x1, y1, x2, y2)
            
        Returns:
            IOU entre 0 e 1
        """
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calcular interseção
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calcular união
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _is_box_completely_outside(self, box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> bool:
        """
        Verifica se box1 está COMPLETAMENTE FORA de box2 (OTIMIZADO).
        
        Uma box está completamente fora quando não há interseção alguma.
        Versão otimizada usando early exit e comparações diretas.
        
        Args:
            box1: (x1, y1, x2, y2) - bounding box a verificar (ex: smudge)
            box2: (x1, y1, x2, y2) - bounding box de referência (ex: outra classe)
            
        Returns:
            True se box1 está completamente fora de box2, False caso contrário
        """
        # Otimização: usar unpacking direto e comparações rápidas
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Early exit: se box1 está completamente separada em qualquer direção, retorna True
        # Ordem otimizada: verificar direções mais prováveis primeiro
        return (x2_1 <= x1_2 or  # Completamente à esquerda
                x1_1 >= x2_2 or  # Completamente à direita
                y2_1 <= y1_2 or  # Completamente acima
                y1_1 >= y2_2)    # Completamente abaixo
    
    def _filter_overlapping_detections(self, detections_by_class: Dict[str, List]) -> Dict[str, List]:
        """
        Remove detecções sobrepostas entre classes, com exclusão mútua para FIFA, Símbolo e String.
        
        REGRA CRÍTICA: Objetos Smudge (FIFA do modelo smudge) são exibidos APENAS se estiverem 
        COMPLETAMENTE FORA das bounding boxes de TODAS as outras classes de detecção (blackdot, simbolos, etc).
        
        - A bounding box de Smudge deve estar completamente separada (sem interseção e sem proximidade)
        - Verificação mais rigorosa que IOU = 0 - garante separação total
        - Não há exceções para esta regra
        - A verificação é aplicada ANTES de qualquer outra regra de exclusão mútua
        
        Args:
            detections_by_class: Dict com detecções por classe
            
        Returns:
            Detecções filtradas (smudge removido se não estiver completamente fora das bounding boxes das outras classes)
        """
        filtered_detections = {class_name: [] for class_name in detections_by_class.keys()}
        
        # Classes que têm exclusão mútua (FIFA, Símbolo, String)
        exclusive_classes = ['smudge', 'simbolos', 'blackdot']  # Classes internas
        
        # Coletar todas as detecções com suas classes
        all_detections = []
        for class_name, detections in detections_by_class.items():
            for detection in detections:
                class_id = detection.get('class_id')
                # Identificar se é FIFA do modelo de símbolos (classes 0-1) ou FIFA (smudge)
                # No modelo best.pt: 0=FIFA_NO, 1=FIFA_OK, 2=Simbolo_NO, 3=Simbolo_OK, 4=String_NO, 5=String_OK
                is_fifa_in_simbolos = (class_name == 'simbolos' and class_id is not None and class_id in [0, 1])  # FIFA_NO ou FIFA_OK do modelo simbolos
                is_fifa_smudge = (class_name == 'smudge')  # FIFA do modelo smudge
                
                # Calcular prioridade baseada na classe
                base_priority = self.class_priority.index(class_name) if class_name in self.class_priority else 999
                
                # Ajustar prioridade:
                # - FIFA do modelo simbolos (classes 0-1): prioridade 1 (equivalente a simbolos)
                # - FIFA do modelo smudge: mantém prioridade 2 (menor, como smudge)
                # - Simbolo e String do modelo simbolos: prioridade 1 (já têm)
                # - Blackdot: prioridade 0 (maior) - já está correto
                if is_fifa_in_simbolos:
                    # FIFA do modelo simbolos tem prioridade equivalente a simbolos (índice 1)
                    priority = 1
                elif is_fifa_smudge:
                    # FIFA do modelo smudge mantém prioridade de smudge (menor)
                    priority = base_priority  # Será 2 (smudge)
                else:
                    priority = base_priority
                
                all_detections.append({
                    'class': class_name,
                    'bbox': detection['bbox'],
                    'confidence': detection['confidence'],
                    'class_id': class_id,  # Preservar class_id (importante para classes OK/NO)
                    'priority': priority,
                    'is_exclusive': class_name in exclusive_classes,
                    'is_fifa_in_simbolos': is_fifa_in_simbolos,  # Marcar se é FIFA do modelo simbolos
                    'is_fifa_smudge': is_fifa_smudge  # Marcar se é FIFA do modelo smudge
                })
        
        # Ordenar por prioridade (menor índice = maior prioridade)
        # Prioridade: blackdot (0) > simbolos/FIFA do simbolos (1) > smudge/FIFA do smudge (2)
        # blackdot, FIFA do simbolos, Simbolo e String têm prioridade sobre smudge
        all_detections.sort(key=lambda x: (x['priority'], -x['confidence']))
        
        # OTIMIZAÇÃO: Criar lista flat de todas as outras detecções (não-smudge) para verificação rápida
        other_detections_flat = []
        for other_class_name, other_detections_list in detections_by_class.items():
            if other_class_name != 'smudge':
                for other_det in other_detections_list:
                    other_detections_flat.append(other_det['bbox'])
        
        # Filtrar sobreposições com exclusão mútua
        # REGRA CRÍTICA: smudge só é exibido se estiver COMPLETAMENTE FORA das bounding boxes de TODAS as outras classes
        for detection in all_detections:
            is_overlapping = False
            
            # OTIMIZAÇÃO: Verificação rápida para smudge - uma única passada
            if detection['is_fifa_smudge']:
                # Verificar contra lista flat de outras detecções (mais rápido que loops aninhados)
                for other_bbox in other_detections_flat:
                    if not self._is_box_completely_outside(detection['bbox'], other_bbox):
                        is_overlapping = True
                        break
                
                # Se smudge foi marcado como overlapping, pular para próxima detecção
                if is_overlapping:
                    continue
            
            # Verificar sobreposição com detecções já aceitas (apenas para classes não-smudge)
            for accepted_class, accepted_detections in filtered_detections.items():
                for accepted_detection in accepted_detections:
                    # EXCEÇÃO: FIFA do modelo smudge e FIFA do modelo simbolos podem coexistir
                    accepted_class_id = accepted_detection.get('class_id')
                    accepted_is_fifa_in_simbolos = (accepted_class == 'simbolos' and accepted_class_id is not None and accepted_class_id in [0, 1])
                    
                    fifa_fifa_intersection = (
                        (detection['is_fifa_smudge'] and accepted_is_fifa_in_simbolos) or
                        (detection.get('is_fifa_in_simbolos') and accepted_class == 'smudge')
                    )
                    
                    if fifa_fifa_intersection:
                        continue
                    
                    # Para classes exclusivas, usar threshold rigoroso
                    if detection['is_exclusive'] and accepted_class != detection['class']:
                        iou = self._calculate_iou(detection['bbox'], accepted_detection['bbox'])
                        if iou > 0.1:
                            is_overlapping = True
                            break
                    # Para outras classes, usar threshold normal
                    elif not detection['is_exclusive']:
                        iou = self._calculate_iou(detection['bbox'], accepted_detection['bbox'])
                        if iou > self.overlap_threshold:
                            is_overlapping = True
                            break
                
                if is_overlapping:
                    break
            
            # Se não há sobreposição, adicionar à lista filtrada
            if not is_overlapping:
                filtered_detection = {
                    'bbox': detection['bbox'],
                    'confidence': detection['confidence']
                }
                # Preservar class_id se existir (para classes do modelo simbolos: FIFA, Simbolo, String com OK/NO)
                if detection.get('class_id') is not None:
                    filtered_detection['class_id'] = detection['class_id']
                # Preservar flags para verificação posterior se necessário
                filtered_detections[detection['class']].append(filtered_detection)
        
        return filtered_detections
    
    def _apply_exclusive_filtering(self, detections_by_class: Dict[str, List]) -> Dict[str, List]:
        """
        Aplica filtro de exclusão mútua com regra especial para Smudge.
        
        REGRA CRÍTICA: Objetos Smudge (FIFA do modelo smudge) são exibidos APENAS se estiverem 
        COMPLETAMENTE FORA das bounding boxes de TODAS as outras classes de detecção.
        
        Sistema otimizado para reduzir conflitos entre classes.
        
        Args:
            detections_by_class: Dict com detecções por classe
            
        Returns:
            Detecções com exclusão mútua aplicada
        """
        # Classes exclusivas (FIFA, Símbolo, String)
        exclusive_classes = ['smudge', 'simbolos', 'blackdot']
        
        # Coletar todas as detecções exclusivas com informações adicionais
        exclusive_detections = []
        for class_name in exclusive_classes:
            if class_name in detections_by_class:
                for detection in detections_by_class[class_name]:
                    class_id = detection.get('class_id')
                    # Identificar se é FIFA do modelo de símbolos (classes 0-1) ou FIFA (smudge)
                    # No modelo best.pt: 0=FIFA_NO, 1=FIFA_OK, 2=Simbolo_NO, 3=Simbolo_OK, 4=String_NO, 5=String_OK
                    is_fifa_in_simbolos = (class_name == 'simbolos' and class_id is not None and class_id in [0, 1])  # FIFA_NO ou FIFA_OK do modelo simbolos
                    is_fifa_smudge = (class_name == 'smudge')  # FIFA do modelo smudge
                    
                    # Calcular prioridade baseada na classe
                    base_priority = self.class_priority.index(class_name) if class_name in self.class_priority else 999
                    
                    # Ajustar prioridade:
                    # - FIFA do modelo simbolos (classes 0-1): prioridade 1 (equivalente a simbolos)
                    # - FIFA do modelo smudge: mantém prioridade 2 (menor, como smudge)
                    # - Simbolo e String do modelo simbolos: prioridade 1 (já têm)
                    # - Blackdot: prioridade 0 (maior) - já está correto
                    if is_fifa_in_simbolos:
                        # FIFA do modelo simbolos tem prioridade equivalente a simbolos (índice 1)
                        priority = 1
                    elif is_fifa_smudge:
                        # FIFA do modelo smudge mantém prioridade de smudge (menor)
                        priority = base_priority  # Será 2 (smudge)
                    else:
                        priority = base_priority
                    
                    exclusive_detections.append({
                        'class': class_name,
                        'bbox': detection['bbox'],
                        'confidence': detection['confidence'],
                        'class_id': class_id,  # Preservar class_id (importante para classes OK/NO)
                        'priority': priority,
                        'is_fifa_in_simbolos': is_fifa_in_simbolos,  # Marcar se é FIFA do modelo simbolos
                        'is_fifa_smudge': is_fifa_smudge  # Marcar se é FIFA do modelo smudge
                    })
        
        # Ordenar por prioridade PRIMEIRO, depois por confiança
        # Prioridade: blackdot (0) > simbolos/FIFA do simbolos (1) > smudge/FIFA do smudge (2)
        # blackdot, FIFA do simbolos, Simbolo e String têm prioridade sobre smudge
        exclusive_detections.sort(key=lambda x: (x['priority'], -x['confidence']))
        
        # OTIMIZAÇÃO: Criar lista flat de outras detecções (não-smudge) para verificação rápida
        other_exclusive_flat = []
        for other_det in exclusive_detections:
            if not other_det.get('is_fifa_smudge', False):
                other_exclusive_flat.append(other_det['bbox'])
        
        # Aplicar exclusão mútua com threshold rigoroso
        # REGRA CRÍTICA: smudge só é exibido se estiver COMPLETAMENTE FORA das bounding boxes de TODAS as outras classes
        filtered_exclusive = []
        for detection in exclusive_detections:
            is_overlapping = False
            
            # OTIMIZAÇÃO: Verificação rápida para smudge
            if detection['is_fifa_smudge']:
                # Verificar contra lista flat (mais rápido)
                for other_bbox in other_exclusive_flat:
                    if not self._is_box_completely_outside(detection['bbox'], other_bbox):
                        is_overlapping = True
                        break
                
                if is_overlapping:
                    continue
            
            # Verificar sobreposição com detecções já aceitas
            for accepted in filtered_exclusive:
                # EXCEÇÃO: FIFA do modelo smudge e FIFA do modelo simbolos podem coexistir
                fifa_fifa_intersection = (
                    (detection['is_fifa_smudge'] and accepted.get('is_fifa_in_simbolos')) or
                    (detection.get('is_fifa_in_simbolos') and accepted.get('is_fifa_smudge'))
                )
                
                if fifa_fifa_intersection:
                    continue
                
                # Para outras classes, usar threshold de IOU
                iou = self._calculate_iou(detection['bbox'], accepted['bbox'])
                if iou > 0.02:
                    is_overlapping = True
                    break
            
            if not is_overlapping:
                filtered_exclusive.append(detection)
        
        # Reconstruir dicionário de detecções
        result = {class_name: [] for class_name in detections_by_class.keys()}
        
        # Adicionar detecções exclusivas filtradas (preservar class_id se existir)
        for detection in filtered_exclusive:
            filtered_detection = {
                'bbox': detection['bbox'],
                'confidence': detection['confidence']
            }
            # Preservar class_id se existir (para classes do modelo simbolos como R_OK e R_NO)
            if detection.get('class_id') is not None:
                filtered_detection['class_id'] = detection['class_id']
            result[detection['class']].append(filtered_detection)
        
        return result
    
    def _stabilize_detection_count(self, class_name: str, current_count: int) -> int:
        """
        Aplica estabilização temporal no contador de detecções com filtro mais robusto.
        
        Args:
            class_name: Nome da classe
            current_count: Contagem atual
            
        Returns:
            Contagem estabilizada
        """
        # Adicionar contagem atual ao histórico
        self.detection_history[class_name].append(current_count)
        
        # Manter apenas os últimos N frames
        if len(self.detection_history[class_name]) > self.stabilization_window:
            self.detection_history[class_name] = self.detection_history[class_name][-self.stabilization_window:]
        
        # Calcular média móvel ponderada (frames mais recentes têm mais peso)
        if len(self.detection_history[class_name]) > 0:
            weights = [i + 1 for i in range(len(self.detection_history[class_name]))]
            weighted_sum = sum(count * weight for count, weight in zip(self.detection_history[class_name], weights))
            total_weight = sum(weights)
            stabilized_count = weighted_sum / total_weight
            
            # Aplicar filtro de mudança brusca (máximo 50% de mudança por frame)
            if len(self.detection_history[class_name]) > 1:
                previous_count = self.detection_history[class_name][-2]
                max_change = max(1, previous_count * 0.5)  # Máximo 50% de mudança
                if abs(stabilized_count - previous_count) > max_change:
                    stabilized_count = previous_count + (1 if stabilized_count > previous_count else -1)
            
            return int(round(stabilized_count))
        
        return current_count
    
    def _validate_detection_quality(self, result, min_confidence: float) -> bool:
        """
        Valida a qualidade de uma detecção.
        
        Args:
            result: Resultado do YOLO
            min_confidence: Confiança mínima
            
        Returns:
            True se a detecção é válida
        """
        if not result or not hasattr(result, 'boxes') or result.boxes is None:
            return False
        
        if len(result.boxes) == 0:
            return False
        
        # Verificar se pelo menos uma detecção tem confiança suficiente
        confidences = result.boxes.conf.cpu().numpy()
        return np.max(confidences) >= min_confidence
    
    def _validate_fifa_detection(self, result, min_confidence: float = 0.7) -> bool:
        """
        Valida detecções de FIFA com critérios RIGOROSOS para reduzir conflitos e falsos positivos.
        
        Args:
            result: Resultado do YOLO para FIFA
            min_confidence: Confiança mínima (padrão 0.7 - aumentado)
            
        Returns:
            True se a detecção de FIFA é válida
        """
        if not result or not hasattr(result, 'boxes') or result.boxes is None:
            return False
        
        if len(result.boxes) == 0:
            return False
        
        # Verificar confiança mínima AUMENTADA para reduzir conflitos
        confidences = result.boxes.conf.cpu().numpy()
        max_conf = np.max(confidences)
        if max_conf < min_confidence:
            return False
        
        # Verificar tamanho das bounding boxes com critérios mais rigorosos
        boxes = result.boxes.xyxy.cpu().numpy()
        valid_boxes = 0
        
        for box in boxes:
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1
            
            # Critérios mais rigorosos para evitar conflitos
            # Rejeitar bounding boxes muito pequenas (aumentado de 20 para 25)
            if width < 25 or height < 25:
                continue
            
            # Rejeitar bounding boxes muito grandes (reduzido de 200 para 150)
            if width > 150 or height > 150:
                continue
            
            # Verificar proporção da bounding box (evitar formas muito alongadas)
            aspect_ratio = width / height
            if aspect_ratio < 0.3 or aspect_ratio > 3.0:
                continue
            
            valid_boxes += 1
        
        # Pelo menos uma bounding box deve ser válida
        if valid_boxes == 0:
            return False
        
        # Log de validação a cada 120 frames com informações de conflito
        if self.frame_count % 120 == 0:
            self.logger.info(f"🔍 FIFA Validation: {len(boxes)} total, {valid_boxes} valid, max_conf={max_conf:.3f}")
        
        return True
    
    def _validate_bbox(self, bbox: Tuple[int, int, int, int], frame_shape: Tuple[int, int]) -> bool:
        """
        Valida se uma bounding box está dentro dos limites do frame e tem tamanho mínimo.
        
        Args:
            bbox: (x1, y1, x2, y2)
            frame_shape: (height, width, channels)
            
        Returns:
            True se a bbox é válida
        """
        x1, y1, x2, y2 = bbox
        height, width = frame_shape[:2]
        
        # Verificar limites do frame
        if x1 < 0 or y1 < 0 or x2 >= width or y2 >= height:
            return False
        
        # Verificar se x2 > x1 e y2 > y1
        if x2 <= x1 or y2 <= y1:
            return False
        
        # Verificar tamanho mínimo (pelo menos 10x10 pixels)
        if (x2 - x1) < 10 or (y2 - y1) < 10:
            return False
        
        # Verificar tamanho máximo (não pode ser maior que 80% do frame)
        max_width = width * 0.8
        max_height = height * 0.8
        if (x2 - x1) > max_width or (y2 - y1) > max_height:
            return False
        
        return True
    
    def _filter_valid_bboxes(self, bboxes: List[Tuple[int, int, int, int]], frame_shape: Tuple[int, int]) -> List[Tuple[int, int, int, int]]:
        """
        Filtra bounding boxes válidas.
        
        Args:
            bboxes: Lista de bounding boxes
            frame_shape: (height, width, channels)
            
        Returns:
            Lista de bounding boxes válidas
        """
        valid_bboxes = []
        for bbox in bboxes:
            if self._validate_bbox(bbox, frame_shape):
                valid_bboxes.append(bbox)
            else:
                self.logger.debug(f"Bbox inválida removida: {bbox}")
        
        return valid_bboxes
    
    def save_parameters(self, config_path: str = "config/last_settings.yaml"):
        """
        Salva os parâmetros atuais do detector para persistência.
        
        Args:
            config_path: Caminho para salvar os parâmetros
        """
        try:
            import yaml
            from pathlib import Path
            
            # Criar diretório se não existir
            Path(config_path).parent.mkdir(exist_ok=True)
            
            # Coletar parâmetros atuais
            parameters = {
                "thresholds": {
                    "roi_conf": self.roi_conf,
                    "roi_iou": self.roi_iou,
                    "smudge_conf": self.smudge_conf,
                    "smudge_iou": self.smudge_iou,
                    "simbolo_conf": self.simbolo_conf,
                    "simbolo_iou": self.simbolo_iou,
                    "blackdot_conf": self.blackdot_conf,
                    "blackdot_iou": self.blackdot_iou
                },
                "model_enabled": self.model_enabled.copy(),
                "inference_params": {
                    "imgsz": self.imgsz,
                    "max_det": self.max_det,
                    "stabilization_window": self.stabilization_window,
                    "min_detection_confidence": self.min_detection_confidence,
                    "overlap_threshold": self.overlap_threshold,
                    "class_priority": self.class_priority.copy()
                },
                "transfer_params": {
                    "absent_to_new": self.ABSENT_TO_NEW,
                    "iou_new_thresh": self.IOU_NEW_THRESH
                }
            }
            
            # Salvar arquivo
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(parameters, f, default_flow_style=False, allow_unicode=True)
            
            self.logger.info(f"✓ Parâmetros salvos em: {config_path}")
            
        except Exception as e:
            self.logger.error(f"Erro ao salvar parâmetros: {e}")
    
    def load_parameters(self, config_path: str = "config/last_settings.yaml"):
        """
        Carrega parâmetros salvos do detector.
        
        Args:
            config_path: Caminho para carregar os parâmetros
            
        Returns:
            True se carregou com sucesso
        """
        try:
            import yaml
            from pathlib import Path
            
            if not Path(config_path).exists():
                self.logger.info("Arquivo de parâmetros não encontrado, usando padrões")
                return False
            
            # Carregar arquivo
            with open(config_path, 'r', encoding='utf-8') as f:
                parameters = yaml.safe_load(f)
            
            if not parameters:
                return False
            
            # Aplicar thresholds
            if "thresholds" in parameters:
                thresholds = parameters["thresholds"]
                self.roi_conf = thresholds.get("roi_conf", self.roi_conf)
                self.roi_iou = thresholds.get("roi_iou", self.roi_iou)
                self.smudge_conf = thresholds.get("smudge_conf", self.smudge_conf)
                self.smudge_iou = thresholds.get("smudge_iou", self.smudge_iou)
                self.simbolo_conf = thresholds.get("simbolo_conf", self.simbolo_conf)
                self.simbolo_iou = thresholds.get("simbolo_iou", self.simbolo_iou)
                self.blackdot_conf = thresholds.get("blackdot_conf", self.blackdot_conf)
                self.blackdot_iou = thresholds.get("blackdot_iou", self.blackdot_iou)
            
            # Aplicar modelos habilitados
            if "model_enabled" in parameters:
                self.model_enabled.update(parameters["model_enabled"])
            
            # Aplicar parâmetros de inferência
            if "inference_params" in parameters:
                inf_params = parameters["inference_params"]
                self.stabilization_window = inf_params.get("stabilization_window", self.stabilization_window)
                self.min_detection_confidence = inf_params.get("min_detection_confidence", self.min_detection_confidence)
                self.overlap_threshold = inf_params.get("overlap_threshold", self.overlap_threshold)
                self.class_priority = inf_params.get("class_priority", self.class_priority)
            
            # Aplicar parâmetros de transfer
            if "transfer_params" in parameters:
                transfer_params = parameters["transfer_params"]
                self.ABSENT_TO_NEW = transfer_params.get("absent_to_new", self.ABSENT_TO_NEW)
                self.IOU_NEW_THRESH = transfer_params.get("iou_new_thresh", self.IOU_NEW_THRESH)
            
            self.logger.info(f"✓ Parâmetros carregados de: {config_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Erro ao carregar parâmetros: {e}")
            return False
    
    def update_thresholds(self, thresholds: Dict[str, float]):
        """
        Atualiza thresholds do detector e salva automaticamente.
        
        Args:
            thresholds: Dicionário com novos thresholds
        """
        try:
            # Atualizar thresholds
            if "roi_conf" in thresholds:
                self.roi_conf = max(0.1, min(1.0, thresholds["roi_conf"]))
            if "roi_iou" in thresholds:
                self.roi_iou = max(0.1, min(1.0, thresholds["roi_iou"]))
            if "smudge_conf" in thresholds:
                self.smudge_conf = max(0.1, min(1.0, thresholds["smudge_conf"]))
            if "smudge_iou" in thresholds:
                self.smudge_iou = max(0.1, min(1.0, thresholds["smudge_iou"]))
            if "simbolo_conf" in thresholds:
                self.simbolo_conf = max(0.1, min(1.0, thresholds["simbolo_conf"]))
            if "simbolo_iou" in thresholds:
                self.simbolo_iou = max(0.1, min(1.0, thresholds["simbolo_iou"]))
            if "blackdot_conf" in thresholds:
                self.blackdot_conf = max(0.1, min(1.0, thresholds["blackdot_conf"]))
            if "blackdot_iou" in thresholds:
                self.blackdot_iou = max(0.1, min(1.0, thresholds["blackdot_iou"]))
            
            # Salvar automaticamente
            self.save_parameters()
            
            self.logger.info(f"✓ Thresholds atualizados e salvos")
            
        except Exception as e:
            self.logger.error(f"Erro ao atualizar thresholds: {e}")
    
    def combine_seg_masks_to_full(self, mask_list, full_shape):
        """Combina máscaras de segmentação em uma máscara completa (baseado no código MacBook)."""
        if mask_list is None or len(mask_list) == 0:
            return None
        
        # Converter para numpy se necessário
        m = mask_list
        if hasattr(m, 'cpu'):
            m = m.cpu().numpy()
        
        # Combinar máscaras com threshold 0.5
        comb = (np.max(m, axis=0) > 0.5).astype(np.uint8) * 255
        H, W = full_shape[:2]
        comb = cv2.resize(comb, (W, H), interpolation=cv2.INTER_NEAREST)
        
        # Operações morfológicas para limpar a máscara
        kernel = np.ones((3, 3), np.uint8)
        comb = cv2.morphologyEx(comb, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        return comb

    def largest_bbox_from_mask(self, mask, margin, frame_shape):
        """Encontra o maior bounding box da máscara (baseado no código MacBook)."""
        h, w = frame_shape[:2]
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None
        
        c = max(cnts, key=cv2.contourArea)
        x, y, bw, bh = cv2.boundingRect(c)
        x = max(0, x - margin)
        y = max(0, y - margin)
        bw = min(w - x, bw + 2 * margin)
        bh = min(h - y, bh + 2 * margin)
        return (x, y, bw, bh)

    def extract_roi_from_segmentation(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]], Optional[np.ndarray], Optional[float]]:
        """
        Extrai ROI a partir da segmentação usando a MÁSCARA (não bbox) - código MacBook.
        
        Returns:
            Tuple (roi_crop, bbox, mask, confidence) onde:
            - roi_crop: crop do frame na área do bbox
            - bbox: (x, y, w, h) no espaço do frame original
            - mask: máscara da ROI no frame completo
            - confidence: confiança da detecção ROI
        """
        if self.seg_model is None or not self.model_enabled["seg"]:
            if not self.model_enabled["seg"]:
                self.logger.debug("Modelo de segmentação ROI desativado")
            else:
                self.logger.error("✗ Modelo de segmentação ROI não está carregado!")
            return None, None, None, None
        
        try:
            # Tamanho original do frame
            orig_h, orig_w = frame.shape[:2]
            
            # Log a cada 60 frames para não poluir
            if self.frame_count % 60 == 0:
                self.logger.debug(f"Executando segmentação ROI no frame {orig_w}x{orig_h} com conf={self.roi_conf:.2f}")
            
            results = self.seg_model.predict(
                frame,
                imgsz=self.imgsz,
                conf=self.roi_conf,
                iou=self.roi_iou,
                verbose=False
            )
            
            if len(results) == 0 or results[0].masks is None:
                if self.frame_count % 60 == 0:
                    self.logger.warning(f"⚠ Nenhuma máscara ROI detectada (conf={self.roi_conf:.2f})")
                return None, None, None, None
            
            # Obter máscaras - USAR MÁSCARA DIRETAMENTE (código MacBook)
            masks = results[0].masks.data
            if len(masks) == 0:
                if self.frame_count % 60 == 0:
                    self.logger.warning(f"⚠ Lista de máscaras vazia")
                return None, None, None, None
            
            if self.frame_count % 60 == 0:
                self.logger.debug(f"✓ Encontradas {len(masks)} máscaras ROI")
            
            # Capturar confiança da detecção
            confidence = float(results[0].boxes.conf[0]) if len(results[0].boxes.conf) > 0 else 0.0
            
            # Usar método do MacBook para combinar máscaras
            combined_mask = self.combine_seg_masks_to_full(masks, frame.shape)
            
            if combined_mask is None:
                return None, None, None, None
            
            # Verificar área mínima
            area = np.count_nonzero(combined_mask)
            if area < self.roi_min_pixels:
                if self.frame_count % 60 == 0:
                    self.logger.debug(f"⚠ ROI muito pequeno: {area}px < {self.roi_min_pixels}px")
                return None, None, None, None
            
            # Encontrar bbox da máscara (apenas para crop)
            margin = 10
            bbox = self.largest_bbox_from_mask(combined_mask, margin, frame.shape)
            
            if bbox is None:
                return None, None, None, None
            
            x, y, w, h = bbox
            
            # Extrair crop e aplicar máscara para limitar a ROI
            roi_crop = frame[y:y+h, x:x+w].copy()
            
            # Aplicar máscara à ROI para limitar a área de detecção
            roi_mask = combined_mask[y:y+h, x:x+w]
            if roi_mask is not None and roi_mask.shape[:2] == roi_crop.shape[:2]:
                # Aplicar máscara aos canais de cor
                for c in range(roi_crop.shape[2]):
                    roi_crop[:, :, c] = np.where(roi_mask > 0, roi_crop[:, :, c], 0)
            
            if self.frame_count % 60 == 0:
                self.logger.debug(f"✓ ROI extraído: posição=({x},{y}) tamanho={w}x{h} área={area}px confiança={confidence:.3f}")
            
            return roi_crop, bbox, combined_mask, confidence
            
        except Exception as e:
            self.logger.error(f"✗ Erro ao extrair ROI: {e}", exc_info=True)
            return None, None, None, None
    
    def boxes_from_result_in_frame(self, result, x0, y0, crop_shape, frame_shape, debug=False):
        """Converte coordenadas do resultado YOLO para o frame original (EXATAMENTE como MacBook)."""
        if result is None or result.boxes is None:
            return np.zeros((0, 4), dtype=int)
        
        # Verificar se há detecções de forma segura
        try:
            num_boxes = result.boxes.shape[0] if hasattr(result.boxes, 'shape') else 0
        except (AttributeError, TypeError):
            num_boxes = 0
            
        if num_boxes == 0:
            return np.zeros((0, 4), dtype=int)
        
        Hf, Wf = frame_shape[:2]
        Hc, Wc = crop_shape[:2]
        
        # EXATAMENTE como no código MacBook: usar xyxyn (normalizado 0-1)
        b = result.boxes.xyxyn.cpu().numpy().astype(float)
        
        if debug and len(b) > 0:
            self.logger.debug(f"   🔄 Conversão bbox: frame={Wf}x{Hf}, crop={Wc}x{Hc}, offset=({x0},{y0})")
            self.logger.debug(f"      Antes (norm): {b[0]}")
        
        # Escalar para o tamanho do crop
        b[:, [0, 2]] *= Wc  # Largura do crop
        b[:, [1, 3]] *= Hc  # Altura do crop
        
        if debug and len(b) > 0:
            self.logger.debug(f"      Após escala crop: {b[0]}")
        
        # Adicionar offset do ROI no frame original
        b[:, [0, 2]] += x0  # Offset X
        b[:, [1, 3]] += y0  # Offset Y
        
        if debug and len(b) > 0:
            self.logger.debug(f"      Após offset: {b[0]}")
        
        # Clipping para limites do frame
        b[:, [0, 2]] = np.clip(b[:, [0, 2]], 0, Wf - 1)
        b[:, [1, 3]] = np.clip(b[:, [1, 3]], 0, Hf - 1)
        
        # Garantir que x1 < x2 e y1 < y2
        x1, y1, x2, y2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
        fix = x2 < x1
        b[fix, 0], b[fix, 2] = x2[fix], x1[fix]
        fix = y2 < y1
        b[fix, 1], b[fix, 3] = y2[fix], y1[fix]
        
        if debug and len(b) > 0:
            self.logger.debug(f"      Final (frame): {b[0].astype(int)}")
        
        # Validar e filtrar bounding boxes
        valid_bboxes = self._filter_valid_bboxes(b.astype(int), frame_shape)
        
        if len(valid_bboxes) == 0:
            return np.array([], dtype=np.int32)
        
        return np.array(valid_bboxes, dtype=np.int32)

    def detect_in_roi(self, roi_crop: np.ndarray, model: YOLO, conf: float, iou: float, model_name: str = "Unknown"):
        """
        Executa detecção em um crop ROI (baseado no código MacBook estável).
        
        Args:
            roi_crop: Imagem crop do ROI
            model: Modelo YOLO para detecção
            conf: Confiança mínima
            iou: IOU threshold
            model_name: Nome do modelo (para logs)
        
        Returns:
            Resultado YOLO completo (para usar com boxes_from_result_in_frame)
        """
        if model is None:
            self.logger.error(f"✗ Modelo {model_name} não está carregado!")
            return None
        
        if roi_crop is None or roi_crop.size == 0:
            return None
        
        try:
            # Log tamanho do crop para debug
            if self.frame_count % 60 == 0:
                self.logger.debug(f"🔍 {model_name}: crop={roi_crop.shape}, conf={conf:.2f}, imgsz={self.imgsz}")
            
            results = model.predict(
                roi_crop,
                imgsz=self.imgsz,
                conf=conf,
                iou=iou,
                max_det=self.max_det,
                verbose=False
            )
            
            if len(results) > 0:
                result = results[0]
                
                # Processamento otimizado - sem melhoria de bounding boxes
                # (Removido para evitar erros e melhorar performance)
                
                # Log detalhado a cada 120 frames (reduzido para melhor performance)
                if self.frame_count % 120 == 0:
                    # Verificar se há detecções de forma segura
                    try:
                        if result.boxes is not None and hasattr(result.boxes, 'shape') and len(result.boxes.shape) > 0:
                            n_boxes = result.boxes.shape[0]
                        else:
                            n_boxes = 0
                    except (AttributeError, TypeError, IndexError):
                        n_boxes = 0
                    if n_boxes > 0:
                        self.logger.info(f"✓ {model_name}: {n_boxes} detecções")
                        # Mostrar primeira bbox em coordenadas normalizadas E absolutas
                        first_box_norm = result.boxes.xyxyn[0].cpu().numpy()
                        first_box_abs = result.boxes.xyxy[0].cpu().numpy()
                        first_conf = float(result.boxes.conf[0].cpu().numpy())
                        self.logger.debug(f"   📐 bbox_norm={first_box_norm}, bbox_abs={first_box_abs}, conf={first_conf:.3f}")
                return result
            
            return None
            
        except Exception as e:
            self.logger.error(f"✗ Erro na detecção {model_name}: {e}")
            return None
    
    # Método _improve_symbol_bboxes removido para otimizar performance
    # e evitar erros de "len() of unsized object"
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Processa um frame completo: ROI + detecções.
        
        Returns:
            Tuple (frame_anotado, estatísticas)
        """
        start_time = time.time()
        
        # Extrair ROI
        # Extrair ROI (agora com máscara - código MacBook)
        roi_crop, roi_bbox, roi_mask, roi_confidence = self.extract_roi_from_segmentation(frame)
        
        annotated_frame = frame.copy()
        stats = {
            "smudge": 0,
            "simbolos": 0,
            "blackdot": 0,
            "has_roi": roi_bbox is not None,
            "roi_confidence": roi_confidence if roi_confidence is not None else 0.0,
            "transfer_count": self.transfer_count,
            "inference_time_ms": 0.0
        }
        
        # Tracking de transfer baseado em ROI
        if roi_bbox is not None:
            # ROI detectado - transfer ativo
            if not self.current_transfer_active:
                # Iniciar novo transfer
                self.current_transfer_active = True
                self.current_transfer_start_frame = self.frame_count
                self.current_transfer_frames = 0
                self.logger.info(f"🔄 Transfer #{self.transfer_count + 1} INICIADO")
            
            # Incrementar frames do transfer atual
            self.current_transfer_frames += 1
            self.frames_without_roi = 0
            self.last_roi_bbox = roi_bbox
            
            # Aplicar suavização na ROI
            smoothed_bbox = self._smooth_bbox(roi_bbox, roi_confidence)
            x, y, w, h = smoothed_bbox
            
            # Desenhar ROI suavizada
            cv2.rectangle(annotated_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Label do ROI com tamanho e confiança
            roi_label = f"ROI {w}x{h} (conf:{roi_confidence:.2f})"
            cv2.putText(annotated_frame, roi_label, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Debug: Log ROI a cada 120 frames (otimizado)
            if self.frame_count % 120 == 0:
                self.logger.info(f"📦 ROI: pos=({x},{y}), tamanho={w}x{h}, crop_shape={roi_crop.shape}, confiança={roi_confidence:.3f}")
            
            # Detectar smudge com validação e estabilização (apenas se modelo ativado)
            if self.model_enabled["smudge"]:
                smudge_result = self.detect_in_roi(roi_crop, self.smudge_model, self.smudge_conf, self.smudge_iou, "FIFA")
                
                # Validar qualidade da detecção FIFA com critérios RIGOROSOS para evitar conflitos
                if self._validate_fifa_detection(smudge_result, min_confidence=0.7):
                    smudge_count = len(smudge_result.boxes) if smudge_result and smudge_result.boxes is not None else 0
                else:
                    smudge_result = None
                    smudge_count = 0
                
                # Aplicar estabilização temporal
                smudge_count = self._stabilize_detection_count("smudge", smudge_count)
            else:
                smudge_result = None
                smudge_count = 0
            stats["smudge"] = smudge_count
            self.current_transfer_stats["smudge"].append(smudge_count)
            
            # Detectar símbolos com validação e estabilização (apenas se modelo ativado)
            if self.model_enabled["simbolos"]:
                simbolos_result = self.detect_in_roi(roi_crop, self.simbolos_model, self.simbolo_conf, self.simbolo_iou, "Símbolos")
                
                # Validar qualidade da detecção
                if self._validate_detection_quality(simbolos_result, self.min_detection_confidence):
                    simbolos_count = len(simbolos_result.boxes) if simbolos_result and simbolos_result.boxes is not None else 0
                else:
                    simbolos_result = None
                    simbolos_count = 0
                
                # Aplicar estabilização temporal
                simbolos_count = self._stabilize_detection_count("simbolos", simbolos_count)
            else:
                simbolos_result = None
                simbolos_count = 0
            stats["simbolos"] = simbolos_count
            self.current_transfer_stats["simbolos"].append(simbolos_count)
            
            # Detectar blackdot com validação e estabilização (apenas se modelo ativado)
            if self.model_enabled["blackdot"]:
                blackdot_result = self.detect_in_roi(roi_crop, self.blackdot_model, self.blackdot_conf, self.blackdot_iou, "BlackDot")
                
                # Validar qualidade da detecção
                if self._validate_detection_quality(blackdot_result, self.min_detection_confidence):
                    blackdot_count = len(blackdot_result.boxes) if blackdot_result and blackdot_result.boxes is not None else 0
                else:
                    blackdot_result = None
                    blackdot_count = 0
                
                # Aplicar estabilização temporal
                blackdot_count = self._stabilize_detection_count("blackdot", blackdot_count)
            else:
                blackdot_result = None
                blackdot_count = 0
            stats["blackdot"] = blackdot_count
            self.current_transfer_stats["blackdot"].append(blackdot_count)
            
            # Aplicar filtros de sobreposição entre classes com exclusão mútua
            detections_by_class = {}
            
            # Coletar detecções válidas e filtrar apenas as que estão dentro da ROI
            # OTIMIZAÇÃO: Converter confianças uma vez em vez de uma por uma
            if smudge_result and smudge_result.boxes is not None and len(smudge_result.boxes) > 0:
                detections_by_class["smudge"] = []
                boxes_coords = self.boxes_from_result_in_frame(smudge_result, x, y, roi_crop.shape, frame.shape, debug=False)
                # Converter todas as confianças de uma vez (mais eficiente)
                confidences = smudge_result.boxes.conf.cpu().numpy()
                for i, coords in enumerate(boxes_coords):
                    # Verificar se a detecção está dentro da ROI
                    if self.is_detection_inside_roi(coords, roi_mask, roi_bbox):
                        detections_by_class["smudge"].append({
                            'bbox': coords,
                            'confidence': float(confidences[i])
                        })
            
            if simbolos_result and simbolos_result.boxes is not None and len(simbolos_result.boxes) > 0:
                detections_by_class["simbolos"] = []
                boxes_coords = self.boxes_from_result_in_frame(simbolos_result, x, y, roi_crop.shape, frame.shape, debug=False)
                # Converter todas as confianças e classes de uma vez (mais eficiente)
                confidences = simbolos_result.boxes.conf.cpu().numpy()
                class_ids = simbolos_result.boxes.cls.cpu().numpy()
                for i, coords in enumerate(boxes_coords):
                    # Verificar se a detecção está dentro da ROI
                    if self.is_detection_inside_roi(coords, roi_mask, roi_bbox):
                        detections_by_class["simbolos"].append({
                            'bbox': coords,
                            'confidence': float(confidences[i]),
                            'class_id': int(class_ids[i])  # Preservar class_id para classes OK/NO (FIFA, Simbolo, String)
                        })
            
            if blackdot_result and blackdot_result.boxes is not None and len(blackdot_result.boxes) > 0:
                detections_by_class["blackdot"] = []
                boxes_coords = self.boxes_from_result_in_frame(blackdot_result, x, y, roi_crop.shape, frame.shape, debug=False)
                # Converter todas as confianças de uma vez (mais eficiente)
                confidences = blackdot_result.boxes.conf.cpu().numpy()
                for i, coords in enumerate(boxes_coords):
                    # Verificar se a detecção está dentro da ROI
                    if self.is_detection_inside_roi(coords, roi_mask, roi_bbox):
                        detections_by_class["blackdot"].append({
                            'bbox': coords,
                            'confidence': float(confidences[i])
                        })
            
            # Aplicar filtros de sobreposição entre classes com exclusão mútua
            if detections_by_class:
                # Aplicar exclusão mútua rigorosa
                filtered_detections = self._apply_exclusive_filtering(detections_by_class)
                
                # Depois aplicar filtro normal de sobreposição
                filtered_detections = self._filter_overlapping_detections(filtered_detections)
                
                # Log de exclusão mútua a cada 120 frames com detalhes de conflitos
                if self.frame_count % 120 == 0:
                    total_before = sum(len(detections) for detections in detections_by_class.values())
                    total_after = sum(len(detections) for detections in filtered_detections.values())
                    if total_before > total_after:
                        conflicts_resolved = total_before - total_after
                        self.logger.info(f"🔒 Exclusão Mútua: {total_before} → {total_after} detecções ({conflicts_resolved} conflitos resolvidos)")
                        
                        # Log detalhado das classes detectadas
                        class_counts = {class_name: len(detections) for class_name, detections in filtered_detections.items() if len(detections) > 0}
                        if class_counts:
                            class_info = ", ".join([f"{self.class_mapping.get(k, k)}:{v}" for k, v in class_counts.items()])
                            self.logger.info(f"   📊 Classes finais: {class_info}")
                
                # Atualizar contagens baseadas nas detecções filtradas
                stats["smudge"] = len(filtered_detections.get("smudge", []))
                stats["simbolos"] = len(filtered_detections.get("simbolos", []))
                stats["blackdot"] = len(filtered_detections.get("blackdot", []))
                
                # Atualizar estatísticas do transfer
                self.current_transfer_stats["smudge"][-1] = stats["smudge"]
                self.current_transfer_stats["simbolos"][-1] = stats["simbolos"]
                self.current_transfer_stats["blackdot"][-1] = stats["blackdot"]
            
            # Desenhar detecções filtradas (já aplicado exclusão mútua acima)
            if detections_by_class:
                
                # Desenhar FIFA filtrado
                for detection in filtered_detections.get("smudge", []):
                    x1, y1, x2, y2 = detection['bbox']
                    conf_val = detection['confidence']
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    label = f"Smudge {conf_val:.2f}"
                    cv2.putText(annotated_frame, label, (x1, max(y1-5, 10)),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                
                # Desenhar símbolos filtrados com nomes corretos das classes
                for detection in filtered_detections.get("simbolos", []):
                    x1, y1, x2, y2 = detection['bbox']
                    conf_val = detection['confidence']
                    
                    # Obter nome da classe usando class_id preservado
                    class_name = "Símbolo"  # Padrão
                    class_id = detection.get('class_id', -1)  # Usar class_id preservado
                    # Classes do modelo best.pt: ['FIFA_NO', 'FIFA_OK', 'Simbolo_NO', 'Simbolo_OK', 'String_NO', 'String_OK'] - 6 classes
                    class_names = ['FIFA_NO', 'FIFA_OK', 'Simbolo_NO', 'Simbolo_OK', 'String_NO', 'String_OK']
                    if 0 <= class_id < len(class_names):
                        class_name = class_names[class_id]
                    
                    # Cor baseada no tipo de classe - cores destacadas
                    # NO: Vermelho destacado (0, 0, 255) em BGR
                    # OK: Verde destacado (0, 255, 0) em BGR
                    if 'OK' in class_name:
                        color = (0, 255, 0)  # Verde destacado para OK (BGR)
                    else:
                        color = (0, 0, 255)  # Vermelho destacado para NO (BGR)
                    
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Label com nome correto da classe - cores destacadas
                    label = f"{class_name} {conf_val:.2f}"
                    # Usar espessura maior para destacar (2 em vez de 1)
                    cv2.putText(annotated_frame, label, (x1, max(y1-5, 10)),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Desenhar String filtrado
                for detection in filtered_detections.get("blackdot", []):
                    x1, y1, x2, y2 = detection['bbox']
                    conf_val = detection['confidence']
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    label = f"BlackDot {conf_val:.2f}"
                    cv2.putText(annotated_frame, label, (x1, max(y1-5, 10)),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        else:
            # ROI não detectado
            self.frames_without_roi += 1
            
            # Se há um transfer ativo e ROI ausente por tempo suficiente, finalizar
            if self.current_transfer_active and self.frames_without_roi >= self.ABSENT_TO_NEW:
                self.logger.info(f"🔄 Transfer #{self.transfer_count + 1} FINALIZADO (ROI ausente por {self.frames_without_roi} frames)")
                self._finalize_transfer()
                self.transfer_count += 1
                self.current_transfer_active = False
                self.current_transfer_frames = 0
                self.last_roi_bbox = None
        
        # Calcular tempo de inferência
        inference_time = (time.time() - start_time) * 1000
        stats["inference_time_ms"] = inference_time
        
        # Atualizar média
        self.frame_count += 1
        alpha = 0.1
        self.avg_inference_time = alpha * inference_time + (1 - alpha) * self.avg_inference_time
        self.last_inference_time = inference_time
        
        # Adicionar médias ao stats
        stats["avg_smudge"] = self.transfer_averages["smudge"]
        stats["avg_simbolos"] = self.transfer_averages["simbolos"]
        stats["avg_blackdot"] = self.transfer_averages["blackdot"]
        
        # Adicionar estatísticas de transfer
        stats["total_evaluated"] = self.transfer_stats["total_evaluated"]
        stats["total_approved"] = self.transfer_stats["total_approved"]
        stats["total_rejected"] = self.transfer_stats["total_rejected"]
        
        # Calcular taxa de aprovação
        total_evaluated = self.transfer_stats["total_evaluated"]
        if total_evaluated > 0:
            stats["approval_rate"] = (self.transfer_stats["total_approved"] / total_evaluated) * 100
        else:
            stats["approval_rate"] = 0.0
        
        # Adicionar estatísticas de classes detectadas médias
        stats["avg_smudge_detected"] = self.transfer_averages.get("smudge", 0.0) * 100  # Em percentual
        stats["avg_simbolos_detected"] = self.transfer_averages.get("simbolos", 0.0) * 100  # Em percentual
        stats["avg_blackdot_detected"] = self.transfer_averages.get("blackdot", 0.0) * 100  # Em percentual
        
        # Calcular classe predominante baseada na média móvel
        predominant_class, predominant_confidence = self._calculate_predominant_class(stats)
        self.predominant_class = predominant_class
        self.predominant_class_confidence = predominant_confidence
        
        # Adicionar informações da classe predominante às estatísticas
        stats["predominant_class"] = predominant_class
        stats["predominant_class_confidence"] = predominant_confidence
        
        # Analisar classes específicas OK/NO se há detecções
        ok_no_stats = {}
        if stats.get("simbolos", 0) > 0 and simbolos_result:
            ok_no_stats = self._analyze_ok_no_classes(simbolos_result)
            stats.update(ok_no_stats)
            
            # Acumular contagens OK/NO por transfer
            if self.current_transfer_active:
                self.current_transfer_stats["fifa_ok"].append(ok_no_stats.get("fifa_ok", 0))
                self.current_transfer_stats["fifa_no"].append(ok_no_stats.get("fifa_no", 0))
                self.current_transfer_stats["simbolo_ok"].append(ok_no_stats.get("simbolo_ok", 0))
                self.current_transfer_stats["simbolo_no"].append(ok_no_stats.get("simbolo_no", 0))
                self.current_transfer_stats["string_ok"].append(ok_no_stats.get("string_ok", 0))
                self.current_transfer_stats["string_no"].append(ok_no_stats.get("string_no", 0))
        else:
            # Se não há detecções, adicionar zeros
            if self.current_transfer_active:
                self.current_transfer_stats["fifa_ok"].append(0)
                self.current_transfer_stats["fifa_no"].append(0)
                self.current_transfer_stats["simbolo_ok"].append(0)
                self.current_transfer_stats["simbolo_no"].append(0)
                self.current_transfer_stats["string_ok"].append(0)
                self.current_transfer_stats["string_no"].append(0)
        
        # Log da classe predominante a cada 60 frames para não poluir
        if self.frame_count % 60 == 0 and predominant_class != "Nenhuma":
            self.logger.info(f"🎯 Classe Predominante: {predominant_class} (confiança: {predominant_confidence:.2f})")
        
        return annotated_frame, stats
    
    def _finalize_transfer(self):
        """Finaliza um transfer e calcula médias e status de aprovação."""
        # Calcular médias e totais
        for key in ["smudge", "simbolos", "blackdot"]:
            if len(self.current_transfer_stats[key]) > 0:
                self.transfer_averages[key] = np.mean(self.current_transfer_stats[key])
            else:
                self.transfer_averages[key] = 0.0
        
        # Calcular totais de objetos detectados por classe
        smudge_total = sum(self.current_transfer_stats["smudge"]) if self.current_transfer_stats["smudge"] else 0
        blackdot_total = sum(self.current_transfer_stats["blackdot"]) if self.current_transfer_stats["blackdot"] else 0
        
        # Calcular totais de OK/NO
        fifa_ok_total = sum(self.current_transfer_stats["fifa_ok"]) if self.current_transfer_stats["fifa_ok"] else 0
        fifa_no_total = sum(self.current_transfer_stats["fifa_no"]) if self.current_transfer_stats["fifa_no"] else 0
        simbolo_ok_total = sum(self.current_transfer_stats["simbolo_ok"]) if self.current_transfer_stats["simbolo_ok"] else 0
        simbolo_no_total = sum(self.current_transfer_stats["simbolo_no"]) if self.current_transfer_stats["simbolo_no"] else 0
        string_ok_total = sum(self.current_transfer_stats["string_ok"]) if self.current_transfer_stats["string_ok"] else 0
        string_no_total = sum(self.current_transfer_stats["string_no"]) if self.current_transfer_stats["string_no"] else 0
        
        # Totais para FIFA, Simbolo e String
        fifa_total = fifa_ok_total + fifa_no_total
        simbolo_total = simbolo_ok_total + simbolo_no_total
        string_total = string_ok_total + string_no_total
        
        # Calcular frames com detecção (pelo menos 1 objeto detectado)
        smudge_frames = sum(1 for count in self.current_transfer_stats["smudge"] if count > 0)
        blackdot_frames = sum(1 for count in self.current_transfer_stats["blackdot"] if count > 0)
        fifa_frames = sum(1 for ok, no in zip(self.current_transfer_stats["fifa_ok"], self.current_transfer_stats["fifa_no"]) if ok > 0 or no > 0)
        simbolo_frames = sum(1 for ok, no in zip(self.current_transfer_stats["simbolo_ok"], self.current_transfer_stats["simbolo_no"]) if ok > 0 or no > 0)
        string_frames = sum(1 for ok, no in zip(self.current_transfer_stats["string_ok"], self.current_transfer_stats["string_no"]) if ok > 0 or no > 0)
        
        # Informações do transfer finalizado
        transfer_duration = self.current_transfer_frames
        transfer_start = self.current_transfer_start_frame
        
        # Determinar se o transfer foi aprovado ou reprovado
        # REGRA: Se qualquer classe for detectada por mais de 20% do transfer, REPROVAR
        
        smudge_avg = self.transfer_averages.get('smudge', 0.0)
        simbolos_avg = self.transfer_averages.get('simbolos', 0.0)
        blackdot_avg = self.transfer_averages.get('blackdot', 0.0)
        
        # Critérios de reprovação: qualquer classe > 20% = REPROVADO
        smudge_reject = smudge_avg > 0.20  # Smudge > 20% = reprovado
        simbolos_reject = simbolos_avg > 0.20  # Símbolos > 20% = reprovado  
        blackdot_reject = blackdot_avg > 0.20  # BlackDot > 20% = reprovado
        
        # Transfer reprovado se QUALQUER classe exceder 20%
        rejected = smudge_reject or simbolos_reject or blackdot_reject
        approved = not rejected
        
        # Atualizar estatísticas
        self.transfer_stats["total_evaluated"] += 1
        if approved:
            self.transfer_stats["total_approved"] += 1
            status = "APROVADO"
        else:
            self.transfer_stats["total_rejected"] += 1
            status = "REPROVADO"
        
        # Adicionar ao histórico com informações detalhadas de objetos detectados
        transfer_record = {
            "transfer_id": self.transfer_count,
            "duration_frames": transfer_duration,
            "start_frame": transfer_start,
            "smudge_avg": smudge_avg,
            "simbolos_avg": simbolos_avg,
            "blackdot_avg": blackdot_avg,
            "approved": approved,
            "classes_detected": {
                "smudge_detected": smudge_avg > 0,
                "simbolos_detected": simbolos_avg > 0,
                "blackdot_detected": blackdot_avg > 0
            },
            "rejection_reasons": {
                "smudge_reject": smudge_reject,
                "simbolos_reject": simbolos_reject,
                "blackdot_reject": blackdot_reject
            },
            "objects_detected": {
                "blackdot": {
                    "total": blackdot_total,
                    "frames_with_detection": blackdot_frames
                },
                "smudge": {
                    "total": smudge_total,
                    "frames_with_detection": smudge_frames
                },
                "fifa": {
                    "total": fifa_total,
                    "ok": fifa_ok_total,
                    "no": fifa_no_total,
                    "frames_with_detection": fifa_frames
                },
                "simbolo": {
                    "total": simbolo_total,
                    "ok": simbolo_ok_total,
                    "no": simbolo_no_total,
                    "frames_with_detection": simbolo_frames
                },
                "string": {
                    "total": string_total,
                    "ok": string_ok_total,
                    "no": string_no_total,
                    "frames_with_detection": string_frames
                }
            }
        }
        
        # Limpar estatísticas do transfer atual
        for key in self.current_transfer_stats:
            self.current_transfer_stats[key] = []
        self.transfer_stats["transfer_history"].append(transfer_record)
        
        # Manter apenas os últimos 100 transfers no histórico
        if len(self.transfer_stats["transfer_history"]) > 100:
            self.transfer_stats["transfer_history"].pop(0)
        
        # Log detalhado com razões de reprovação
        rejection_details = []
        if smudge_reject:
            rejection_details.append(f"Smudge({smudge_avg:.1%})")
        if simbolos_reject:
            rejection_details.append(f"Símbolos({simbolos_avg:.1%})")
        if blackdot_reject:
            rejection_details.append(f"BlackDot({blackdot_avg:.1%})")
        
        rejection_info = f" - Razões: {', '.join(rejection_details)}" if rejection_details else ""
        
        self.logger.info(f"Transfer #{self.transfer_count} finalizado - "
                        f"Frames: {transfer_duration}, "
                        f"Smudge: {smudge_avg:.1%}, "
                        f"Símbolos: {simbolos_avg:.1%}, "
                        f"BlackDot: {blackdot_avg:.1%} - "
                        f"STATUS: {status}{rejection_info}")
        
        # Log das estatísticas gerais
        total = self.transfer_stats["total_evaluated"]
        approved_count = self.transfer_stats["total_approved"]
        rejected_count = self.transfer_stats["total_rejected"]
        approval_rate = (approved_count / total * 100) if total > 0 else 0
        
        self.logger.info(f"Estatísticas Gerais - Total: {total}, "
                        f"Aprovados: {approved_count}, "
                        f"Reprovados: {rejected_count}, "
                        f"Taxa Aprovação: {approval_rate:.1f}%")
    
    def get_performance_stats(self) -> dict:
        """Retorna estatísticas de performance."""
        return {
            "avg_inference_ms": self.avg_inference_time,
            "last_inference_ms": self.last_inference_time,
            "frame_count": self.frame_count,
            "transfer_count": self.transfer_count,
            "device": self.device
        }
    
    def get_transfer_class_stats(self) -> dict:
        """Retorna estatísticas das classes detectadas por transfer."""
        if not self.transfer_stats["transfer_history"]:
            return {
                "total_transfers": 0,
                "smudge_detected_count": 0,
                "simbolos_detected_count": 0,
                "blackdot_detected_count": 0,
                "smudge_detection_rate": 0.0,
                "simbolos_detection_rate": 0.0,
                "blackdot_detection_rate": 0.0
            }
        
        total_transfers = len(self.transfer_stats["transfer_history"])
        smudge_detected = sum(1 for t in self.transfer_stats["transfer_history"] if t["classes_detected"]["smudge_detected"])
        simbolos_detected = sum(1 for t in self.transfer_stats["transfer_history"] if t["classes_detected"]["simbolos_detected"])
        blackdot_detected = sum(1 for t in self.transfer_stats["transfer_history"] if t["classes_detected"]["blackdot_detected"])
        
        return {
            "total_transfers": total_transfers,
            "smudge_detected_count": smudge_detected,
            "simbolos_detected_count": simbolos_detected,
            "blackdot_detected_count": blackdot_detected,
            "smudge_detection_rate": (smudge_detected / total_transfers) * 100,
            "simbolos_detection_rate": (simbolos_detected / total_transfers) * 100,
            "blackdot_detection_rate": (blackdot_detected / total_transfers) * 100
        }
    
    def get_final_statistics_summary(self) -> dict:
        """
        Gera sumário final de estatísticas de todos os transfers avaliados.
        
        Returns:
            Dicionário com sumário completo das estatísticas
        """
        transfer_history = self.transfer_stats["transfer_history"]
        total_transfers = len(transfer_history)
        
        if total_transfers == 0:
            return {
                "total_transfers_evaluated": 0,
                "transfers_by_class": {},
                "most_frequent_error": {
                    "class": "Nenhuma",
                    "type": "N/A",
                    "count": 0,
                    "percentage": 0.0
                },
                "objects_per_transfer": []
            }
        
        # Contabilizar totais por classe
        blackdot_total = 0
        smudge_total = 0
        fifa_total = 0
        fifa_ok_total = 0
        fifa_no_total = 0
        simbolo_total = 0
        simbolo_ok_total = 0
        simbolo_no_total = 0
        string_total = 0
        string_ok_total = 0
        string_no_total = 0
        
        # Contabilizar quantos transfers tiveram cada classe
        blackdot_count = 0
        smudge_count = 0
        fifa_count = 0
        simbolo_count = 0
        string_count = 0
        
        # Lista de objetos por transfer
        objects_per_transfer = []
        
        # Processar cada transfer no histórico
        for transfer in transfer_history:
            objects_detected = transfer.get("objects_detected", {})
            
            # Blackdot
            blackdot_obj = objects_detected.get("blackdot", {})
            blackdot_transfer_total = blackdot_obj.get("total", 0)
            blackdot_total += blackdot_transfer_total
            if blackdot_transfer_total > 0:
                blackdot_count += 1
            
            # Smudge
            smudge_obj = objects_detected.get("smudge", {})
            smudge_transfer_total = smudge_obj.get("total", 0)
            smudge_total += smudge_transfer_total
            if smudge_transfer_total > 0:
                smudge_count += 1
            
            # FIFA
            fifa_obj = objects_detected.get("fifa", {})
            fifa_transfer_total = fifa_obj.get("total", 0)
            fifa_transfer_ok = fifa_obj.get("ok", 0)
            fifa_transfer_no = fifa_obj.get("no", 0)
            fifa_total += fifa_transfer_total
            fifa_ok_total += fifa_transfer_ok
            fifa_no_total += fifa_transfer_no
            if fifa_transfer_total > 0:
                fifa_count += 1
            
            # Simbolo
            simbolo_obj = objects_detected.get("simbolo", {})
            simbolo_transfer_total = simbolo_obj.get("total", 0)
            simbolo_transfer_ok = simbolo_obj.get("ok", 0)
            simbolo_transfer_no = simbolo_obj.get("no", 0)
            simbolo_total += simbolo_transfer_total
            simbolo_ok_total += simbolo_transfer_ok
            simbolo_no_total += simbolo_transfer_no
            if simbolo_transfer_total > 0:
                simbolo_count += 1
            
            # String
            string_obj = objects_detected.get("string", {})
            string_transfer_total = string_obj.get("total", 0)
            string_transfer_ok = string_obj.get("ok", 0)
            string_transfer_no = string_obj.get("no", 0)
            string_total += string_transfer_total
            string_ok_total += string_transfer_ok
            string_no_total += string_transfer_no
            if string_transfer_total > 0:
                string_count += 1
            
            # Adicionar à lista de objetos por transfer
            objects_per_transfer.append({
                "transfer_id": transfer.get("transfer_id", 0),
                "blackdot": blackdot_transfer_total,
                "smudge": smudge_transfer_total,
                "fifa": {
                    "ok": fifa_transfer_ok,
                    "no": fifa_transfer_no
                },
                "simbolo": {
                    "ok": simbolo_transfer_ok,
                    "no": simbolo_transfer_no
                },
                "string": {
                    "ok": string_transfer_ok,
                    "no": string_transfer_no
                }
            })
        
        # Construir sumário por classe
        transfers_by_class = {
            "blackdot": {
                "count": blackdot_count,
                "total_objects": blackdot_total
            },
            "smudge": {
                "count": smudge_count,
                "total_objects": smudge_total
            },
            "fifa": {
                "count": fifa_count,
                "total_objects": fifa_total,
                "ok": fifa_ok_total,
                "no": fifa_no_total
            },
            "simbolo": {
                "count": simbolo_count,
                "total_objects": simbolo_total,
                "ok": simbolo_ok_total,
                "no": simbolo_no_total
            },
            "string": {
                "count": string_count,
                "total_objects": string_total,
                "ok": string_ok_total,
                "no": string_no_total
            }
        }
        
        # Determinar erro mais frequente
        # Erros são classes NO (FIFA_NO, Simbolo_NO, String_NO) ou classes sem OK/NO quando detectadas (smudge, blackdot)
        error_counts = {
            "fifa_no": fifa_no_total,
            "simbolo_no": simbolo_no_total,
            "string_no": string_no_total,
            "smudge": smudge_total,  # Smudge sempre é considerado erro se detectado
            "blackdot": blackdot_total  # Blackdot sempre é considerado erro se detectado
        }
        
        # Encontrar o erro mais frequente
        most_frequent_error_class = max(error_counts, key=error_counts.get)
        most_frequent_error_count = error_counts[most_frequent_error_class]
        
        # Calcular percentual
        total_errors = sum(error_counts.values())
        most_frequent_error_percentage = (most_frequent_error_count / total_errors * 100) if total_errors > 0 else 0.0
        
        # Mapear nome da classe para exibição
        class_name_map = {
            "fifa_no": "FIFA",
            "simbolo_no": "Simbolo",
            "string_no": "String",
            "smudge": "Smudge",
            "blackdot": "BlackDot"
        }
        
        most_frequent_error = {
            "class": class_name_map.get(most_frequent_error_class, most_frequent_error_class),
            "type": "NO" if "_no" in most_frequent_error_class else "Erro",
            "count": most_frequent_error_count,
            "percentage": most_frequent_error_percentage
        }
        
        return {
            "total_transfers_evaluated": total_transfers,
            "transfers_by_class": transfers_by_class,
            "most_frequent_error": most_frequent_error,
            "objects_per_transfer": objects_per_transfer
        }
    
    def export_statistics_to_file(self, output_dir: str = "logs") -> Optional[str]:
        """
        Exporta estatísticas finais para arquivo JSON.
        
        Args:
            output_dir: Diretório onde salvar o arquivo (padrão: "logs")
            
        Returns:
            Caminho do arquivo criado ou None em caso de erro
        """
        try:
            from pathlib import Path
            import json
            from datetime import datetime
            
            # Criar diretório se não existir
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)
            
            # Obter sumário de estatísticas
            summary = self.get_final_statistics_summary()
            
            # Adicionar informações gerais
            full_stats = {
                "export_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "system_info": {
                    "device": self.device,
                    "total_frames_processed": self.frame_count,
                    "total_transfers": self.transfer_count
                },
                "transfer_statistics": {
                    "total_evaluated": self.transfer_stats["total_evaluated"],
                    "total_approved": self.transfer_stats["total_approved"],
                    "total_rejected": self.transfer_stats["total_rejected"],
                    "approval_rate": (self.transfer_stats["total_approved"] / self.transfer_stats["total_evaluated"] * 100) if self.transfer_stats["total_evaluated"] > 0 else 0.0
                },
                "summary": summary,
                "transfer_history": self.transfer_stats["transfer_history"]
            }
            
            # Nome do arquivo fixo (sobrescrever a cada processamento)
            filename = "statistics.json"
            filepath = output_path / filename
            
            # Salvar arquivo JSON
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(full_stats, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"✓ Estatísticas exportadas para: {filepath}")
            return str(filepath)
            
        except Exception as e:
            self.logger.error(f"Erro ao exportar estatísticas: {e}", exc_info=True)
            return None
    
    def set_model_enabled(self, model_name: str, enabled: bool):
        """Ativa/desativa um modelo específico."""
        if model_name in self.model_enabled:
            self.model_enabled[model_name] = enabled
            self.logger.info(f"Modelo {model_name}: {'ATIVADO' if enabled else 'DESATIVADO'}")
        else:
            self.logger.warning(f"Modelo {model_name} não encontrado")
    
    def get_model_status(self) -> dict:
        """Retorna status de ativação dos modelos."""
        return self.model_enabled.copy()
    
    def enable_all_models(self):
        """Ativa todos os modelos."""
        for model in self.model_enabled:
            self.model_enabled[model] = True
        self.logger.info("Todos os modelos ATIVADOS")
    
    def disable_all_models(self):
        """Desativa todos os modelos."""
        for model in self.model_enabled:
            self.model_enabled[model] = False
        self.logger.info("Todos os modelos DESATIVADOS")
    
    def _stabilize_smudge_detection(self, smudge_result, smudge_count):
        """
        Estabiliza detecções de smudge para reduzir falsos positivos/negativos.
        
        Args:
            smudge_result: Resultado da detecção atual
            smudge_count: Número de detecções atual
            
        Returns:
            Tuple (resultado_estabilizado, count_estabilizado)
        """
        try:
            # Adicionar detecção atual ao histórico
            self.smudge_history.append({
                'result': smudge_result,
                'count': smudge_count,
                'frame': self.frame_count
            })
            
            # Manter apenas os últimos N frames
            max_history = 10
            if len(self.smudge_history) > max_history:
                self.smudge_history = self.smudge_history[-max_history:]
            
            # Se não há detecções suficientes para estabilizar
            if len(self.smudge_history) < self.smudge_stability_threshold:
                return smudge_result, smudge_count
            
            # Analisar padrão de detecções recentes
            recent_detections = self.smudge_history[-self.smudge_stability_threshold:]
            detection_counts = [d['count'] for d in recent_detections]
            
            # Calcular estabilidade
            avg_count = sum(detection_counts) / len(detection_counts)
            variance = sum((c - avg_count) ** 2 for c in detection_counts) / len(detection_counts)
            
            # Se a variância é baixa (detecções estáveis)
            if variance < 0.5:  # Threshold de estabilidade
                # Usar detecção mais recente se há consenso
                if avg_count > 0.5:  # Maioria dos frames tem detecção
                    # Retornar a detecção mais recente com confiança
                    latest_detection = recent_detections[-1]
                    self.smudge_stable_detection = latest_detection['result']
                    return latest_detection['result'], int(avg_count + 0.5)
                else:
                    # Consenso de não detecção
                    self.smudge_stable_detection = None
                    return None, 0
            else:
                # Detecções instáveis - manter detecção estável anterior
                if self.smudge_stable_detection is not None:
                    # Verificar se a detecção estável ainda é válida
                    stable_age = self.frame_count - (self.smudge_history[-1]['frame'] if self.smudge_history else 0)
                    if stable_age < 5:  # Detecção estável ainda recente
                        return self.smudge_stable_detection, 1
                
                # Se não há detecção estável ou é muito antiga, usar atual
                return smudge_result, smudge_count
                
        except Exception as e:
            self.logger.warning(f"Erro na estabilização de smudge: {e}")
            return smudge_result, smudge_count
    
    def _stabilize_symbols_detection(self, symbols_result, symbols_count):
        """
        Estabiliza detecções de símbolos para reduzir falsos positivos/negativos.
        
        Args:
            symbols_result: Resultado da detecção atual
            symbols_count: Número de detecções atual
            
        Returns:
            Tuple (resultado_estabilizado, count_estabilizado)
        """
        try:
            # Adicionar detecção atual ao histórico
            self.symbols_history.append({
                'result': symbols_result,
                'count': symbols_count,
                'frame': self.frame_count
            })
            
            # Manter apenas os últimos N frames
            max_history = 10
            if len(self.symbols_history) > max_history:
                self.symbols_history = self.symbols_history[-max_history:]
            
            # Se não há detecções suficientes para estabilizar
            if len(self.symbols_history) < self.symbols_stability_threshold:
                return symbols_result, symbols_count
            
            # Analisar padrão de detecções recentes
            recent_detections = self.symbols_history[-self.symbols_stability_threshold:]
            detection_counts = [d['count'] for d in recent_detections]
            
            # Calcular estabilidade
            avg_count = sum(detection_counts) / len(detection_counts)
            variance = sum((c - avg_count) ** 2 for c in detection_counts) / len(detection_counts)
            
            # Se a variância é baixa (detecções estáveis)
            if variance < 0.5:  # Threshold de estabilidade
                # Usar detecção mais recente se há consenso
                if avg_count > 0.5:  # Maioria dos frames tem detecção
                    # Retornar a detecção mais recente com confiança
                    latest_detection = recent_detections[-1]
                    self.symbols_stable_detection = latest_detection['result']
                    return latest_detection['result'], int(avg_count + 0.5)
                else:
                    # Consenso de não detecção
                    self.symbols_stable_detection = None
                    return None, 0
            else:
                # Detecções instáveis - manter detecção estável anterior
                if self.symbols_stable_detection is not None:
                    # Verificar se a detecção estável ainda é válida
                    stable_age = self.frame_count - (self.symbols_history[-1]['frame'] if self.symbols_history else 0)
                    if stable_age < 5:  # Detecção estável ainda recente
                        return self.symbols_stable_detection, 1
                
                # Se não há detecção estável ou é muito antiga, usar atual
                return symbols_result, symbols_count
                
        except Exception as e:
            self.logger.warning(f"Erro na estabilização de símbolos: {e}")
            return symbols_result, symbols_count
    
    def _stabilize_class_detection(self, result, class_name, history_key):
        """
        Estabiliza detecções de classes específicas (FIFA, String).
        
        Args:
            result: Resultado da detecção atual
            class_name: Nome da classe (FIFA, String)
            history_key: Chave do histórico (fifa_history, string_history)
            
        Returns:
            Tuple (resultado_estabilizado, count_estabilizado)
        """
        try:
            history = getattr(self, history_key)
            stable_detection_key = f"{class_name.lower()}_stable_detection"
            stable_detection = getattr(self, stable_detection_key)
            
            # Adicionar detecção atual ao histórico
            history.append({
                'result': result,
                'count': len(result.boxes) if result and result.boxes is not None else 0,
                'frame': self.frame_count
            })
            
            # Manter apenas os últimos N frames
            max_history = 8
            if len(history) > max_history:
                history = history[-max_history:]
                setattr(self, history_key, history)
            
            # Se não há detecções suficientes para estabilizar
            if len(history) < 3:
                return result, len(result.boxes) if result and result.boxes is not None else 0
            
            # Analisar padrão de detecções recentes
            recent_detections = history[-3:]
            detection_counts = [d['count'] for d in recent_detections]
            
            # Calcular estabilidade
            avg_count = sum(detection_counts) / len(detection_counts)
            variance = sum((c - avg_count) ** 2 for c in detection_counts) / len(detection_counts)
            
            # Se a variância é baixa (detecções estáveis)
            if variance < 0.3:  # Threshold mais rigoroso para classes específicas
                # Usar detecção mais recente se há consenso
                if avg_count > 0.3:  # Maioria dos frames tem detecção
                    # Retornar a detecção mais recente com confiança
                    latest_detection = recent_detections[-1]
                    setattr(self, stable_detection_key, latest_detection['result'])
                    return latest_detection['result'], int(avg_count + 0.5)
                else:
                    # Consenso de não detecção
                    setattr(self, stable_detection_key, None)
                    return None, 0
            else:
                # Detecções instáveis - manter detecção estável anterior
                if stable_detection is not None:
                    # Verificar se a detecção estável ainda é válida
                    stable_age = self.frame_count - (history[-1]['frame'] if history else 0)
                    if stable_age < 3:  # Detecção estável ainda recente
                        return stable_detection, 1
                
                # Se não há detecção estável ou é muito antiga, usar atual
                return result, len(result.boxes) if result and result.boxes is not None else 0
                
        except Exception as e:
            self.logger.warning(f"Erro na estabilização de {class_name}: {e}")
            return result, len(result.boxes) if result and result.boxes is not None else 0
    
    def _calculate_predominant_class(self, stats: dict) -> tuple:
        """
        Calcula a classe predominante baseada na média móvel das últimas 10 frames.
        
        Args:
            stats: Estatísticas do frame atual
            
        Returns:
            Tuple (classe_predominante, confiança)
        """
        try:
            # Adicionar estatísticas do frame atual ao histórico
            frame_data = {
                'smudge': stats.get('smudge', 0),
                'simbolos': stats.get('simbolos', 0),
                'blackdot': stats.get('blackdot', 0),
                'frame': self.frame_count
            }
            
            self.class_history.append(frame_data)
            
            # Manter apenas os últimos N frames
            if len(self.class_history) > self.moving_average_window:
                self.class_history = self.class_history[-self.moving_average_window:]
            
            # Se não há histórico suficiente, retornar classe atual
            if len(self.class_history) < 3:
                return self._get_current_dominant_class(stats)
            
            # Calcular médias móveis para cada classe
            smudge_avg = sum(f['smudge'] for f in self.class_history) / len(self.class_history)
            simbolos_avg = sum(f['simbolos'] for f in self.class_history) / len(self.class_history)
            blackdot_avg = sum(f['blackdot'] for f in self.class_history) / len(self.class_history)
            
            # Determinar classe predominante baseada nas médias móveis
            class_scores = {
                'Smudge': smudge_avg,
                'Símbolo': simbolos_avg,
                'BlackDot': blackdot_avg
            }
            
            # Encontrar a classe com maior média móvel
            predominant_class = max(class_scores, key=class_scores.get)
            max_score = class_scores[predominant_class]
            
            # Se nenhuma classe tem detecção significativa, retornar "Nenhuma"
            if max_score < 0.1:  # Threshold mínimo para considerar uma classe predominante
                return "Nenhuma", 0.0
            
            # Calcular confiança baseada na estabilidade da detecção
            # Confiança aumenta com a consistência da detecção ao longo dos frames
            recent_frames = self.class_history[-5:] if len(self.class_history) >= 5 else self.class_history
            
            # Contar quantos frames recentes têm a classe predominante detectada
            predominant_detections = 0
            for frame in recent_frames:
                if frame[predominant_class.lower().replace('í', 'i').replace('ó', 'o')] > 0:
                    predominant_detections += 1
            
            # Confiança baseada na frequência de detecção nos frames recentes
            confidence = predominant_detections / len(recent_frames)
            
            # Ajustar confiança baseada na intensidade da detecção
            intensity_factor = min(max_score / 2.0, 1.0)  # Normalizar para 0-1
            final_confidence = confidence * intensity_factor
            
            return predominant_class, final_confidence
            
        except Exception as e:
            self.logger.warning(f"Erro ao calcular classe predominante: {e}")
            return self._get_current_dominant_class(stats)
    
    def _get_current_dominant_class(self, stats: dict) -> tuple:
        """
        Determina a classe dominante do frame atual.
        
        Args:
            stats: Estatísticas do frame atual
            
        Returns:
            Tuple (classe_dominante, confiança)
        """
        smudge_count = stats.get('smudge', 0)
        simbolos_count = stats.get('simbolos', 0)
        blackdot_count = stats.get('blackdot', 0)
        
        # Se nenhuma classe foi detectada
        if smudge_count == 0 and simbolos_count == 0 and blackdot_count == 0:
            return "Nenhuma", 0.0
        
        # Encontrar a classe com maior contagem
        counts = {
            'Smudge': smudge_count,
            'Símbolo': simbolos_count,
            'BlackDot': blackdot_count
        }
        
        dominant_class = max(counts, key=counts.get)
        max_count = counts[dominant_class]
        
        # Calcular confiança baseada na contagem
        total_detections = sum(counts.values())
        confidence = max_count / total_detections if total_detections > 0 else 0.0
        
        return dominant_class, confidence
    
    def _analyze_ok_no_classes(self, simbolos_result) -> dict:
        """
        Analisa as classes específicas OK/NO detectadas.
        
        Args:
            simbolos_result: Resultado da detecção de símbolos
            
        Returns:
            Dicionário com estatísticas das classes OK/NO
        """
        try:
            if not simbolos_result or not simbolos_result.boxes:
                return {
                    "fifa_ok": 0, "fifa_no": 0,
                    "simbolo_ok": 0, "simbolo_no": 0,
                    "string_ok": 0, "string_no": 0,
                    "r_ok": 0, "r_no": 0,  # Mantido para compatibilidade, mas não será usado
                    "total_ok": 0, "total_no": 0
                }
            
            # Obter classes detectadas
            classes = simbolos_result.boxes.cls.cpu().numpy()
            
            # Contar cada classe específica (baseado no modelo best.pt)
            # ['FIFA_NO', 'FIFA_OK', 'Simbolo_NO', 'Simbolo_OK', 'String_NO', 'String_OK'] - 6 classes
            fifa_no = np.sum(classes == 0)    # FIFA_NO
            fifa_ok = np.sum(classes == 1)     # FIFA_OK
            simbolo_no = np.sum(classes == 2)  # Simbolo_NO
            simbolo_ok = np.sum(classes == 3)  # Simbolo_OK
            string_no = np.sum(classes == 4)   # String_NO
            string_ok = np.sum(classes == 5)   # String_OK
            
            # Totais
            total_ok = fifa_ok + simbolo_ok + string_ok
            total_no = fifa_no + simbolo_no + string_no
            
            return {
                "fifa_ok": int(fifa_ok),
                "fifa_no": int(fifa_no),
                "simbolo_ok": int(simbolo_ok),
                "simbolo_no": int(simbolo_no),
                "string_ok": int(string_ok),
                "string_no": int(string_no),
                # Compatibilidade com código existente
                "smudge_ok": int(fifa_ok),  # FIFA é mapeado como smudge
                "smudge_no": int(fifa_no),  # FIFA é mapeado como smudge
                "blackdot_ok": int(string_ok),  # String é mapeado como blackdot
                "blackdot_no": int(string_no),  # String é mapeado como blackdot
                "r_ok": 0,  # Não há mais R no modelo
                "r_no": 0,  # Não há mais R no modelo
                "total_ok": int(total_ok),
                "total_no": int(total_no)
            }
            
        except Exception as e:
            self.logger.warning(f"Erro ao analisar classes OK/NO: {e}")
            return {
                "fifa_ok": 0, "fifa_no": 0,
                "simbolo_ok": 0, "simbolo_no": 0,
                "string_ok": 0, "string_no": 0,
                "r_ok": 0, "r_no": 0,  # Mantido para compatibilidade, mas não será usado
                # Compatibilidade
                "smudge_ok": 0, "smudge_no": 0,
                "blackdot_ok": 0, "blackdot_no": 0,
                "total_ok": 0, "total_no": 0
            }

