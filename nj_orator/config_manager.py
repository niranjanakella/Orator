"""
Configuration management for Orator TTS
Handles loading and saving configuration including hotkeys
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class HotkeyConfig:
    """Hotkey configuration for a single action"""
    type: str  # "single", "combination", "double_tap"
    key: Optional[str] = None
    modifiers: Optional[list] = None
    timeout: Optional[float] = None

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        result = {"type": self.type}
        if self.key:
            result["key"] = self.key
        if self.modifiers:
            result["modifiers"] = self.modifiers
        if self.timeout is not None:
            result["timeout"] = self.timeout
        return result

    @classmethod
    def from_dict(cls, data: dict) -> 'HotkeyConfig':
        """Create from dictionary"""
        return cls(
            type=data.get("type", "single"),
            key=data.get("key"),
            modifiers=data.get("modifiers"),
            timeout=data.get("timeout")
        )


class ConfigManager:
    """Manages Orator configuration including hotkeys"""

    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize config manager
        
        Args:
            config_dir: Directory for config files. If None, uses ~/.orator
        """
        if config_dir is None:
            config_dir = os.path.join(os.path.expanduser("~"), ".orator")
        
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        self.config_file = self.config_dir / "config.json"
        self.pid_file = self.config_dir / "orator.pid"

    def get_app_data_dir(self) -> Path:
        """Get directory for app data (models, voices, etc.)"""
        # Try to find package data directory
        try:
            import nj_orator
            package_dir = Path(nj_orator.__file__).parent.parent
            # Check if kokoro_model_onnx exists in package
            model_dir = package_dir / "kokoro_model_onnx"
            if model_dir.exists():
                return package_dir
        except Exception:
            pass
        
        # Fallback to config directory
        return self.config_dir

    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file"""
        if not self.config_file.exists():
            return self._get_default_config()

        try:
            with open(self.config_file, 'r') as f:
                config = json.load(f)
            
            # Ensure hotkeys structure exists
            if "hotkeys" not in config:
                config["hotkeys"] = {}
            
            return config
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return self._get_default_config()

    def save_config(self, config: Dict[str, Any]) -> bool:
        """Save configuration to file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
            return False

    def get_hotkey(self, action: str) -> Optional[HotkeyConfig]:
        """Get hotkey configuration for an action"""
        config = self.load_config()
        hotkeys = config.get("hotkeys", {})
        
        if action not in hotkeys:
            return None
        
        return HotkeyConfig.from_dict(hotkeys[action])

    def set_hotkey(self, action: str, hotkey: HotkeyConfig) -> bool:
        """Set hotkey configuration for an action"""
        config = self.load_config()
        
        if "hotkeys" not in config:
            config["hotkeys"] = {}
        
        config["hotkeys"][action] = hotkey.to_dict()
        return self.save_config(config)

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        app_data_dir = self.get_app_data_dir()
        
        return {
            "model_path": str(app_data_dir / "kokoro_model_onnx" / "kokoro-v1_0.onnx"),
            "tokenizer_path": str(app_data_dir / "kokoro_model_onnx" / "tokenizer.json"),
            "voices_dir": str(app_data_dir / "kokoro_model_onnx" / "voices"),
            "voice": "bf_isabella",
            "hotkey_timeout": 0.5,
            "max_text_length": None,
            "device": "cpu",
            "speed": 1.0,
            "hotkeys": {
                "trigger": {
                    "type": "double_tap",
                    "key": "alt",
                    "timeout": 0.5
                },
                "pause": {
                    "type": "combination",
                    "modifiers": ["cmd"],
                    "key": "p"
                },
                "stop": {
                    "type": "single",
                    "key": "esc"
                }
            }
        }

    def get_pid_file_path(self) -> Path:
        """Get path to PID file"""
        return self.pid_file

