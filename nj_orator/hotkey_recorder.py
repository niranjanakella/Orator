"""
Interactive hotkey recording for Orator TTS
Captures key presses and records them as hotkey configurations
"""

import time
import logging
from typing import Optional, List, Callable, Dict, Any
from pynput import keyboard
from pynput.keyboard import Key, Listener

logger = logging.getLogger(__name__)


class HotkeyRecorder:
    """Records hotkey combinations interactively"""

    def __init__(self, timeout: float = 5.0):
        """
        Initialize hotkey recorder
        
        Args:
            timeout: Maximum seconds to wait for key press
        """
        self.timeout = timeout
        self.recorded_keys = []
        self.recorded_modifiers = []
        self.modifier_presses = []  # Track modifier key presses with timestamps for double-tap detection
        self.listener = None
        self.start_time = None
        self.recording = False

    def record(self) -> Optional[Dict[str, Any]]:
        """
        Record a hotkey combination
        
        Returns:
            Dictionary with hotkey configuration or None if cancelled/timed out
        """
        print("\n" + "="*60)
        print("Hotkey Recording Mode")
        print("="*60)
        print("Press your desired hotkey combination now...")
        print("(Press ESC to cancel, or wait 5 seconds to timeout)")
        print("="*60 + "\n")

        self.recorded_keys = []
        self.recorded_modifiers = []
        self.modifier_presses = []
        self.start_time = time.time()
        self.recording = True

        try:
            self.listener = Listener(
                on_press=self._on_key_press,
                on_release=self._on_key_release
            )
            self.listener.start()

            # Wait for recording to complete or timeout
            while self.recording:
                if time.time() - self.start_time > self.timeout:
                    print("\nTimeout: No key combination recorded")
                    self.recording = False
                    break
                time.sleep(0.1)

            self.listener.stop()
            self.listener = None

            if not self.recorded_keys and not self.recorded_modifiers:
                return None

            return self._build_hotkey_config()

        except KeyboardInterrupt:
            print("\n\nRecording cancelled by user")
            if self.listener:
                self.listener.stop()
            return None
        except Exception as e:
            logger.error(f"Error during recording: {e}")
            if self.listener:
                self.listener.stop()
            return None

    def _on_key_press(self, key):
        """Handle key press event"""
        if not self.recording:
            return

        # Check for ESC to cancel
        if key == Key.esc:
            print("\n\nRecording cancelled (ESC pressed)")
            self.recording = False
            return False  # Stop listener

        current_time = time.time()
        modifier_name = None

        # Check for modifiers
        if key in [Key.ctrl, Key.ctrl_l, Key.ctrl_r]:
            modifier_name = "ctrl"
            if "ctrl" not in self.recorded_modifiers:
                self.recorded_modifiers.append("ctrl")
        elif key in [Key.alt, Key.alt_l, Key.alt_r]:
            modifier_name = "alt"
            if "alt" not in self.recorded_modifiers:
                self.recorded_modifiers.append("alt")
        elif key in [Key.shift, Key.shift_l, Key.shift_r]:
            modifier_name = "shift"
            if "shift" not in self.recorded_modifiers:
                self.recorded_modifiers.append("shift")
        elif key in [Key.cmd, Key.cmd_l, Key.cmd_r]:
            modifier_name = "cmd"
            if "cmd" not in self.recorded_modifiers:
                self.recorded_modifiers.append("cmd")
        
        # Track modifier presses for double-tap detection
        if modifier_name:
            self.modifier_presses.append((modifier_name, current_time))
            
            # Check for double-tap (same modifier pressed twice within 0.5 seconds)
            if len(self.modifier_presses) >= 2:
                last_two = self.modifier_presses[-2:]
                if last_two[0][0] == last_two[1][0]:  # Same modifier
                    time_diff = last_two[1][1] - last_two[0][1]
                    if time_diff <= 0.5:  # Within 0.5 seconds
                        # Double-tap detected!
                        self.recorded_keys = [modifier_name, modifier_name]  # Mark as double-tap
                        self.recording = False
                        return False  # Stop listener
        
        if not modifier_name:
            # Regular key (not a modifier)
            key_name = self._get_key_name(key)
            if key_name and key_name not in self.recorded_keys:
                self.recorded_keys.append(key_name)
                # If we got a regular key, we can finish recording
                self.recording = False
                return False  # Stop listener

    def _on_key_release(self, key):
        """Handle key release event"""
        # We don't need to do anything on release for single key combinations
        pass

    def _get_key_name(self, key) -> Optional[str]:
        """Convert key object to string name"""
        try:
            if hasattr(key, 'char') and key.char:
                return key.char.lower()
            elif key == Key.space:
                return "space"
            elif key == Key.enter:
                return "enter"
            elif key == Key.tab:
                return "tab"
            elif key == Key.backspace:
                return "backspace"
            elif key == Key.delete:
                return "delete"
            elif key == Key.up:
                return "up"
            elif key == Key.down:
                return "down"
            elif key == Key.left:
                return "left"
            elif key == Key.right:
                return "right"
            elif key == Key.esc:
                return "esc"
            elif key == Key.f1:
                return "f1"
            elif key == Key.f2:
                return "f2"
            elif key == Key.f3:
                return "f3"
            elif key == Key.f4:
                return "f4"
            elif key == Key.f5:
                return "f5"
            elif key == Key.f6:
                return "f6"
            elif key == Key.f7:
                return "f7"
            elif key == Key.f8:
                return "f8"
            elif key == Key.f9:
                return "f9"
            elif key == Key.f10:
                return "f10"
            elif key == Key.f11:
                return "f11"
            elif key == Key.f12:
                return "f12"
            else:
                # Try to get name from key object
                key_str = str(key).replace("Key.", "")
                if key_str:
                    return key_str.lower()
        except Exception:
            pass
        return None

    def _build_hotkey_config(self) -> Optional[Dict[str, Any]]:
        """Build hotkey configuration from recorded keys"""
        if not self.recorded_keys and not self.recorded_modifiers:
            return None

        # Double tap detection (check first - highest priority)
        # This works for both regular keys and modifiers
        if len(self.recorded_keys) == 2 and self.recorded_keys[0] == self.recorded_keys[1]:
            return {
                "type": "double_tap",
                "key": self.recorded_keys[0],
                "timeout": 0.5
            }
        
        # Check modifier double-tap from modifier_presses
        if len(self.modifier_presses) >= 2:
            last_two = self.modifier_presses[-2:]
            if last_two[0][0] == last_two[1][0]:  # Same modifier
                time_diff = last_two[1][1] - last_two[0][1]
                if time_diff <= 0.5:  # Within 0.5 seconds
                    return {
                        "type": "double_tap",
                        "key": last_two[0][0],
                        "timeout": 0.5
                    }

        # Single key press (no modifiers)
        if not self.recorded_modifiers and len(self.recorded_keys) == 1:
            return {
                "type": "single",
                "key": self.recorded_keys[0]
            }

        # Single modifier press (no regular keys)
        if len(self.recorded_modifiers) == 1 and not self.recorded_keys:
            return {
                "type": "single",
                "key": self.recorded_modifiers[0]
            }

        # Key combination (modifiers + key)
        if self.recorded_modifiers and len(self.recorded_keys) == 1:
            return {
                "type": "combination",
                "modifiers": self.recorded_modifiers,
                "key": self.recorded_keys[0]
            }

        # Fallback: combination (only if we have at least a key)
        if self.recorded_keys:
            return {
                "type": "combination",
                "modifiers": self.recorded_modifiers if self.recorded_modifiers else [],
                "key": self.recorded_keys[0]
            }
        
        # No valid keys recorded
        return None

    def format_hotkey_display(self, config: Dict[str, Any]) -> str:
        """Format hotkey configuration for display"""
        if not config:
            return "None"

        hotkey_type = config.get("type", "single")
        
        if hotkey_type == "single":
            key = config.get("key")
            if not key:
                return "Not configured"
            return f"{key.upper()}"
        
        elif hotkey_type == "combination":
            modifiers = config.get("modifiers", [])
            key = config.get("key")
            if not key:
                return "Not configured"
            if not modifiers:
                return f"{key.upper()}"
            mod_str = "+".join([m.upper() for m in modifiers if m])
            return f"{mod_str}+{key.upper()}"
        
        elif hotkey_type == "double_tap":
            key = config.get("key")
            if not key:
                return "Not configured"
            timeout = config.get("timeout", 0.5)
            return f"Double-tap {key.upper()} (within {timeout}s)"
        
        return "Unknown"

