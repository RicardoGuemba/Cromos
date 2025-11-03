"""
Gerenciador de persistência de configurações
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

class ConfigManager:
    """Gerencia persistência de configurações do sistema."""
    
    def __init__(self, config_path: str = "config/app.yaml", settings_path: str = "config/last_settings.yaml"):
        self.config_path = Path(config_path)
        self.settings_path = Path(settings_path)
        self.logger = logging.getLogger(__name__)
        
        # Garantir que o diretório existe
        self.settings_path.parent.mkdir(parents=True, exist_ok=True)
    
    def load_config(self) -> Dict[str, Any]:
        """Carrega configuração principal."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            self.logger.info(f"✓ Configuração carregada: {self.config_path}")
            return config
        except Exception as e:
            self.logger.error(f"Erro ao carregar configuração: {e}")
            return {}
    
    def save_settings(self, settings: Dict[str, Any]) -> bool:
        """Salva configurações atuais."""
        try:
            # Adicionar timestamp
            settings_with_meta = {
                "timestamp": datetime.now().isoformat(),
                "version": "1.0",
                "settings": settings
            }
            
            with open(self.settings_path, 'w', encoding='utf-8') as f:
                yaml.dump(settings_with_meta, f, default_flow_style=False, allow_unicode=True)
            
            self.logger.info(f"✓ Configurações salvas: {self.settings_path}")
            return True
        except Exception as e:
            self.logger.error(f"Erro ao salvar configurações: {e}")
            return False
    
    def load_settings(self) -> Optional[Dict[str, Any]]:
        """Carrega configurações salvas."""
        try:
            if not self.settings_path.exists():
                self.logger.info("Nenhuma configuração salva encontrada")
                return None
            
            with open(self.settings_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            
            if "settings" in data:
                self.logger.info(f"✓ Configurações restauradas: {self.settings_path}")
                return data["settings"]
            else:
                self.logger.warning("Formato de configuração inválido")
                return None
                
        except Exception as e:
            self.logger.error(f"Erro ao carregar configurações salvas: {e}")
            return None
    
    def merge_settings(self, base_config: Dict[str, Any], saved_settings: Dict[str, Any]) -> Dict[str, Any]:
        """Mescla configuração base com configurações salvas."""
        try:
            # Fazer uma cópia profunda da configuração base
            merged_config = self._deep_copy(base_config)
            
            # Aplicar configurações salvas
            self._deep_update(merged_config, saved_settings)
            
            self.logger.info("✓ Configurações mescladas com sucesso")
            return merged_config
            
        except Exception as e:
            self.logger.error(f"Erro ao mesclar configurações: {e}")
            return base_config
    
    def _deep_copy(self, obj):
        """Cópia profunda de dicionário."""
        if isinstance(obj, dict):
            return {key: self._deep_copy(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._deep_copy(item) for item in obj]
        else:
            return obj
    
    def _deep_update(self, base_dict: Dict, update_dict: Dict):
        """Atualização profunda de dicionário."""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def get_ui_settings(self, ui_instance) -> Dict[str, Any]:
        """Extrai configurações da interface."""
        try:
            settings = {
                "camera": {
                    "fps": getattr(ui_instance, 'fps_var', None),
                    "exposure": getattr(ui_instance, 'exposure_var', None),
                    "gain": getattr(ui_instance, 'gain_var', None),
                    "balance": getattr(ui_instance, 'balance_var', None),
                    "resolution": getattr(ui_instance, 'resolution_var', None)
                },
                "thresholds": {
                    "smudge": getattr(ui_instance, 'smudge_conf_var', None),
                    "simbolos": getattr(ui_instance, 'simbolo_conf_var', None),
                    "blackdot": getattr(ui_instance, 'blackdot_conf_var', None)
                },
                "models": {
                    "seg": getattr(ui_instance, 'seg_enabled_var', None),
                    "smudge": getattr(ui_instance, 'smudge_enabled_var', None),
                    "simbolos": getattr(ui_instance, 'simbolos_enabled_var', None),
                    "blackdot": getattr(ui_instance, 'blackdot_enabled_var', None)
                },
                "focus": {
                    "focus": getattr(ui_instance, 'focus_var', None),
                    "auto_focus": getattr(ui_instance, 'auto_focus_var', None)
                }
            }
            
            # Filtrar valores None
            filtered_settings = {}
            for category, values in settings.items():
                filtered_values = {k: v for k, v in values.items() if v is not None}
                if filtered_values:
                    filtered_settings[category] = filtered_values
            
            return filtered_settings
            
        except Exception as e:
            self.logger.error(f"Erro ao extrair configurações da UI: {e}")
            return {}
    
    def apply_ui_settings(self, ui_instance, settings: Dict[str, Any]):
        """Aplica configurações salvas à interface."""
        try:
            # Aplicar configurações da câmera
            if "camera" in settings:
                camera_settings = settings["camera"]
                if "fps" in camera_settings and hasattr(ui_instance, 'fps_var'):
                    ui_instance.fps_var.set(camera_settings["fps"])
                if "exposure" in camera_settings and hasattr(ui_instance, 'exposure_var'):
                    ui_instance.exposure_var.set(camera_settings["exposure"])
                if "gain" in camera_settings and hasattr(ui_instance, 'gain_var'):
                    ui_instance.gain_var.set(camera_settings["gain"])
                if "balance" in camera_settings and hasattr(ui_instance, 'balance_var'):
                    ui_instance.balance_var.set(camera_settings["balance"])
                if "resolution" in camera_settings and hasattr(ui_instance, 'resolution_var'):
                    ui_instance.resolution_var.set(camera_settings["resolution"])
            
            # Aplicar thresholds
            if "thresholds" in settings:
                threshold_settings = settings["thresholds"]
                if "smudge" in threshold_settings and hasattr(ui_instance, 'smudge_conf_var'):
                    ui_instance.smudge_conf_var.set(threshold_settings["smudge"])
                if "simbolos" in threshold_settings and hasattr(ui_instance, 'simbolo_conf_var'):
                    ui_instance.simbolo_conf_var.set(threshold_settings["simbolos"])
                if "blackdot" in threshold_settings and hasattr(ui_instance, 'blackdot_conf_var'):
                    ui_instance.blackdot_conf_var.set(threshold_settings["blackdot"])
            
            # Aplicar modelos
            if "models" in settings:
                model_settings = settings["models"]
                if "seg" in model_settings and hasattr(ui_instance, 'seg_enabled_var'):
                    ui_instance.seg_enabled_var.set(model_settings["seg"])
                if "smudge" in model_settings and hasattr(ui_instance, 'smudge_enabled_var'):
                    ui_instance.smudge_enabled_var.set(model_settings["smudge"])
                if "simbolos" in model_settings and hasattr(ui_instance, 'simbolos_enabled_var'):
                    ui_instance.simbolos_enabled_var.set(model_settings["simbolos"])
                if "blackdot" in model_settings and hasattr(ui_instance, 'blackdot_enabled_var'):
                    ui_instance.blackdot_enabled_var.set(model_settings["blackdot"])
            
            # Aplicar foco
            if "focus" in settings:
                focus_settings = settings["focus"]
                if "focus" in focus_settings and hasattr(ui_instance, 'focus_var'):
                    ui_instance.focus_var.set(focus_settings["focus"])
                if "auto_focus" in focus_settings and hasattr(ui_instance, 'auto_focus_var'):
                    ui_instance.auto_focus_var.set(focus_settings["auto_focus"])
            
            self.logger.info("✓ Configurações aplicadas à interface")
            
        except Exception as e:
            self.logger.error(f"Erro ao aplicar configurações à UI: {e}")
    
    def cleanup_old_settings(self, max_age_days: int = 30):
        """Remove configurações antigas."""
        try:
            if not self.settings_path.exists():
                return
            
            # Verificar idade do arquivo
            file_age = datetime.now().timestamp() - self.settings_path.stat().st_mtime
            age_days = file_age / (24 * 3600)
            
            if age_days > max_age_days:
                self.settings_path.unlink()
                self.logger.info(f"Configurações antigas removidas ({age_days:.1f} dias)")
                
        except Exception as e:
            self.logger.error(f"Erro ao limpar configurações antigas: {e}")
