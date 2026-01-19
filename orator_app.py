#!/usr/bin/env python3
"""
Orator - macOS Menu Bar TTS Application

A menu bar application for text-to-speech using Kokoro with ONNX Runtime.
Features:
- Start/Stop TTS engine
- Voice selection with language grouping
- Status indicator
- Double Option key hotkey for TTS
"""

import os
import sys
import json
import signal
import time
import threading
import queue
import logging
from dataclasses import dataclass
from typing import Optional, Callable
import io

import numpy as np
import rumps
import pygame
import pyperclip
import soundfile as sf
from pynput import keyboard

from orator.onnx_engine import ONNXTTSEngine, AppConfig, AudioData

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def safe_notification(title: str, subtitle: str, message: str) -> None:
    """Send a notification, failing silently if not available."""
    try:
        rumps.notification(title, subtitle, message)
    except Exception:
        # Notifications may fail without Info.plist when running from terminal
        logger.debug(f"Notification skipped: {title} - {subtitle} - {message}")


# Voice language groups for menu organization
VOICE_GROUPS = {
    'American English': ['af_alloy', 'af_aoede', 'af_bella', 'af_heart', 'af_jessica',
                         'af_kore', 'af_nicole', 'af_nova', 'af_river', 'af_sarah', 'af_sky',
                         'am_adam', 'am_echo', 'am_eric', 'am_fenrir', 'am_liam',
                         'am_michael', 'am_onyx', 'am_puck', 'am_santa'],
    'British English': ['bf_alice', 'bf_emma', 'bf_isabella', 'bf_lily',
                        'bm_daniel', 'bm_fable', 'bm_george', 'bm_lewis'],
    'Spanish': ['ef_dora', 'em_alex', 'em_santa'],
    'French': ['ff_siwis'],
    'Hindi': ['hf_alpha', 'hf_beta', 'hm_omega', 'hm_psi'],
    'Italian': ['if_sara', 'im_nicola'],
    'Japanese': ['jf_alpha', 'jf_gongitsune', 'jf_nezumi', 'jf_tebukuro', 'jm_kumo'],
    'Portuguese': ['pf_dora', 'pm_alex', 'pm_santa'],
    'Chinese': ['zf_xiaobei', 'zf_xiaoni', 'zf_xiaoxiao', 'zf_xiaoyi',
                'zm_yunjian', 'zm_yunxi', 'zm_yunxia', 'zm_yunyang'],
}


class AudioPlayer:
    """Audio playback component using pygame"""

    def __init__(self):
        self.mixer_initialized = False
        self.current_sound = None
        self._playback_lock = threading.Lock()
        self._audio_cache = {}
        self._audio_queue = None
        self._streaming_active = False
        self._streaming_thread = None
        self._producer_thread = None
        self._paused = False
        self._overlay_channel_index = 7
        self._stop_event = threading.Event()  # For reliable stop signaling

    def initialize(self) -> bool:
        """Initialize pygame mixer"""
        try:
            pygame.mixer.pre_init(frequency=24000, size=-16, channels=1, buffer=1024)
            pygame.mixer.init()
            try:
                pygame.mixer.set_num_channels(8)
            except Exception:
                pass
            self.mixer_initialized = True
            logger.info("Audio player initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize audio player: {e}")
            return False

    def play_audio(self, audio_data: AudioData) -> bool:
        """Play audio data with thread safety"""
        with self._playback_lock:
            return self._play_audio_internal(audio_data)

    def _play_audio_internal(self, audio_data: AudioData) -> bool:
        """Internal audio playback method"""
        try:
            if not self.mixer_initialized:
                logger.error("Audio player not initialized")
                return False

            self.stop_current()

            audio_samples = audio_data.samples
            if audio_samples.dtype != np.int16:
                max_val = np.max(np.abs(audio_samples))
                if max_val > 1.0:
                    audio_samples = audio_samples / max_val
                audio_samples = (audio_samples * 32767).astype(np.int16)

            buffer = io.BytesIO()
            sf.write(buffer, audio_samples, audio_data.sample_rate, format='WAV')
            buffer.seek(0)

            self.current_sound = pygame.mixer.Sound(buffer)
            self.current_sound.play()

            logger.info(f"Playing audio: {audio_data.duration:.2f}s")
            return True

        except Exception as e:
            logger.error(f"Failed to play audio: {e}")
            return False

    def pause(self) -> bool:
        """Pause playback"""
        try:
            if not self.mixer_initialized or not pygame.mixer.get_busy():
                return False
            pygame.mixer.pause()
            self._paused = True
            logger.info("Audio paused")
            return True
        except Exception as e:
            logger.error(f"Failed to pause audio: {e}")
            return False

    def resume(self) -> bool:
        """Resume playback"""
        try:
            if not self.mixer_initialized:
                return False
            if not self._paused and not pygame.mixer.get_busy():
                return False
            pygame.mixer.unpause()
            self._paused = False
            logger.info("Audio resumed")
            return True
        except Exception as e:
            logger.error(f"Failed to resume audio: {e}")
            return False

    def toggle_pause(self) -> bool:
        """Toggle between pause and resume"""
        return self.resume() if self._paused else self.pause()

    def is_active(self) -> bool:
        """Return True if audio is playing or paused"""
        if not self.mixer_initialized:
            return False
        try:
            return self._paused or pygame.mixer.get_busy()
        except Exception:
            return self._paused

    def is_paused(self) -> bool:
        """Return True if paused"""
        return bool(self._paused)

    def stop_current(self) -> None:
        """Stop current playback immediately"""
        try:
            logger.info("Stopping audio playback...")
            
            # Set stop event and flags immediately
            self._stop_event.set()
            self._streaming_active = False
            self._paused = False

            # Stop all audio playback IMMEDIATELY - don't wait for threads
            if self.mixer_initialized:
                try:
                    pygame.mixer.stop()  # Stop all sounds
                    # Also stop all channels individually
                    for i in range(pygame.mixer.get_num_channels()):
                        try:
                            channel = pygame.mixer.Channel(i)
                            if channel.get_busy():
                                channel.stop()
                        except Exception:
                            pass
                except Exception as e:
                    logger.error(f"Error stopping mixer: {e}")
                
                self.current_sound = None

            # Clear the queue to prevent processing more chunks
            if self._audio_queue:
                try:
                    while not self._audio_queue.empty():
                        try:
                            self._audio_queue.get_nowait()
                        except queue.Empty:
                            break
                except Exception:
                    pass

            logger.info("Audio playback stopped")

        except Exception as e:
            logger.error(f"Error stopping audio: {e}")

    def play_audio_stream(self, audio_chunks_generator) -> bool:
        """Play streaming audio chunks"""
        try:
            if not self.mixer_initialized:
                return False

            # Stop any current playback first
            self.stop_current()

            # Reset stop event and create new queue
            self._stop_event.clear()
            self._audio_queue = queue.Queue(maxsize=10)  # Increased buffer size
            self._streaming_active = True
            self._paused = False

            self._producer_thread = threading.Thread(
                target=self._audio_producer_worker,
                args=(audio_chunks_generator,),
                daemon=True,
                name="AudioProducer"
            )

            self._streaming_thread = threading.Thread(
                target=self._audio_consumer_worker,
                daemon=True,
                name="AudioConsumer"
            )

            self._producer_thread.start()
            self._streaming_thread.start()

            return True

        except Exception as e:
            logger.error(f"Failed to start streaming: {e}")
            return False

    def _audio_producer_worker(self, audio_chunks_generator) -> None:
        """Producer thread for audio chunks"""
        try:
            for audio_chunk in audio_chunks_generator:
                # Check stop event and flag frequently
                if self._stop_event.is_set() or not self._streaming_active:
                    logger.info("Producer: Stop signal received, exiting")
                    break

                audio_data = AudioData(
                    samples=audio_chunk.samples,
                    sample_rate=audio_chunk.sample_rate,
                    duration=len(audio_chunk.samples) / audio_chunk.sample_rate
                )

                try:
                    # Use a shorter timeout and check stop event
                    self._audio_queue.put(
                        (audio_data, audio_chunk.chunk_index, audio_chunk.is_final),
                        timeout=1.0
                    )
                except queue.Full:
                    # If queue is full, check if we should stop
                    if self._stop_event.is_set() or not self._streaming_active:
                        break
                    continue

            # Signal end of stream only if still active
            if not self._stop_event.is_set() and self._streaming_active:
                try:
                    self._audio_queue.put((None, -1, True), timeout=1.0)
                except queue.Full:
                    pass  # Queue full, consumer will detect end when queue empties

        except Exception as e:
            logger.error(f"Audio producer error: {e}")
            # Signal error by putting None in queue only if not stopped
            if not self._stop_event.is_set() and self._streaming_active:
                try:
                    self._audio_queue.put((None, -1, True), timeout=0.5)
                except Exception:
                    pass

    def _audio_consumer_worker(self) -> None:
        """Consumer thread for audio playback with seamless chunk queuing"""
        try:
            # Use a dedicated channel for seamless playback
            channel = None
            try:
                channel = pygame.mixer.Channel(0)
            except Exception:
                pass  # Fallback to default if channel allocation fails
            
            chunks_queued = 0
            
            while not self._stop_event.is_set() and self._streaming_active:
                try:
                    # Use shorter timeout to check stop event more frequently
                    item = self._audio_queue.get(timeout=0.5)

                    # Check stop event immediately after getting item
                    if self._stop_event.is_set() or not self._streaming_active:
                        logger.info("Consumer: Stop signal received, exiting")
                        break

                    if item is None or item[0] is None:
                        break

                    audio_data, chunk_index, is_final = item
                    
                    # Queue chunk immediately for seamless playback
                    if channel:
                        # Queue on channel - pygame will play chunks seamlessly
                        queued = self._play_chunk_on_channel(audio_data, channel)
                        if queued:
                            chunks_queued += 1
                            # Don't wait - queue next chunk immediately
                            # Only pause if explicitly paused
                            if self._paused:
                                while self._paused and not self._stop_event.is_set() and self._streaming_active:
                                    time.sleep(0.01)
                    else:
                        # Fallback: play and wait (less seamless)
                        self._play_chunk_internal(audio_data)
                        if self.current_sound:
                            while pygame.mixer.get_busy() and not self._stop_event.is_set() and self._streaming_active:
                                if self._paused:
                                    time.sleep(0.01)
                                    continue
                                time.sleep(0.01)

                    # Check stop event after processing
                    if self._stop_event.is_set() or not self._streaming_active:
                        break

                    if is_final:
                        # Wait for all queued chunks to finish, but check stop event frequently
                        if channel:
                            while channel.get_busy() and not self._stop_event.is_set() and self._streaming_active:
                                if self._paused:
                                    time.sleep(0.01)
                                    continue
                                time.sleep(0.01)
                        break

                except queue.Empty:
                    # Check stop event while waiting
                    if self._stop_event.is_set() or not self._streaming_active:
                        break
                    continue

            # Ensure we stop playback when exiting
            if self._stop_event.is_set():
                if self.mixer_initialized:
                    try:
                        pygame.mixer.stop()
                        if channel:
                            channel.stop()
                    except Exception:
                        pass

            self._streaming_active = False
            logger.info("Consumer thread exiting")

        except Exception as e:
            logger.error(f"Audio consumer error: {e}")
            self._streaming_active = False
            self._stop_event.set()

    def _play_chunk_internal(self, audio_data: AudioData) -> bool:
        """Play a single chunk"""
        try:
            if not self.mixer_initialized:
                return False

            audio_samples = audio_data.samples
            if audio_samples.dtype != np.int16:
                max_val = np.max(np.abs(audio_samples))
                if max_val > 1.0:
                    audio_samples = audio_samples / max_val
                audio_samples = (audio_samples * 32767).astype(np.int16)

            buffer = io.BytesIO()
            sf.write(buffer, audio_samples, audio_data.sample_rate, format='WAV')
            buffer.seek(0)

            self.current_sound = pygame.mixer.Sound(buffer)
            self.current_sound.play()
            return True

        except Exception as e:
            logger.error(f"Failed to play chunk: {e}")
            return False

    def _play_chunk_on_channel(self, audio_data: AudioData, channel) -> bool:
        """Play a single chunk on a dedicated channel for seamless playback"""
        try:
            if not self.mixer_initialized:
                return False

            audio_samples = audio_data.samples
            if audio_samples.dtype != np.int16:
                max_val = np.max(np.abs(audio_samples))
                if max_val > 1.0:
                    audio_samples = audio_samples / max_val
                audio_samples = (audio_samples * 32767).astype(np.int16)

            buffer = io.BytesIO()
            sf.write(buffer, audio_samples, audio_data.sample_rate, format='WAV')
            buffer.seek(0)

            sound = pygame.mixer.Sound(buffer)
            
            # If channel is not busy, play immediately; otherwise queue for seamless playback
            if not channel.get_busy():
                channel.play(sound)
            else:
                # Queue for seamless playback - pygame will play this after current sound finishes
                channel.queue(sound)
            
            self.current_sound = sound
            return True

        except Exception as e:
            logger.error(f"Failed to play chunk on channel: {e}")
            return False

    def play_notification(self, sound_type: str = "success") -> bool:
        """Play notification sound"""
        try:
            if not self.mixer_initialized:
                return False

            sample_rate = 24000
            duration = 0.2

            if sound_type == "success":
                frequency = 800
            elif sound_type == "error":
                frequency = 400
            elif sound_type == "no_text":
                frequency = 600
            else:
                frequency = 600

            t = np.linspace(0, duration, int(sample_rate * duration))

            if sound_type == "no_text":
                beep1 = np.sin(2 * np.pi * frequency * t[:len(t)//3])
                silence = np.zeros(len(t)//6)
                beep2 = np.sin(2 * np.pi * frequency * t[:len(t)//3])
                padding = np.zeros(len(t) - len(beep1) - len(silence) - len(beep2))
                audio_samples = np.concatenate([beep1, silence, beep2, padding])
            else:
                audio_samples = np.sin(2 * np.pi * frequency * t)

            fade_samples = int(0.01 * sample_rate)
            audio_samples[:fade_samples] *= np.linspace(0, 1, fade_samples)
            audio_samples[-fade_samples:] *= np.linspace(1, 0, fade_samples)
            audio_samples = (audio_samples * 0.3 * 32767).astype(np.int16)

            notification_audio = AudioData(
                samples=audio_samples,
                sample_rate=sample_rate,
                duration=duration
            )

            return self.play_audio(notification_audio)

        except Exception as e:
            logger.error(f"Failed to play notification: {e}")
            return False

    def play_notification_overlay(self, sound_type: str = "success", volume: float = 0.35) -> bool:
        """Play notification without interrupting current playback"""
        try:
            if not self.mixer_initialized:
                return False

            sample_rate = 24000
            duration = 0.2

            if sound_type == "success":
                frequency = 800
            elif sound_type == "error":
                frequency = 400
            else:
                frequency = 600

            t = np.linspace(0, duration, int(sample_rate * duration))
            audio_samples = np.sin(2 * np.pi * frequency * t)

            fade_samples = int(0.01 * sample_rate)
            audio_samples[:fade_samples] *= np.linspace(0, 1, fade_samples)
            audio_samples[-fade_samples:] *= np.linspace(1, 0, fade_samples)
            audio_samples = (audio_samples * 0.3 * 32767).astype(np.int16)

            buffer = io.BytesIO()
            sf.write(buffer, audio_samples, sample_rate, format='WAV')
            buffer.seek(0)

            sound = pygame.mixer.Sound(buffer)
            try:
                channel = pygame.mixer.Channel(self._overlay_channel_index)
                sound.set_volume(max(0.0, min(1.0, volume)))
                channel.play(sound)
                return True
            except Exception:
                sound.play()
                return True

        except Exception as e:
            logger.error(f"Failed to play overlay notification: {e}")
            return False

    def cleanup(self) -> None:
        """Clean up resources"""
        try:
            if self.mixer_initialized:
                self.stop_current()
                pygame.mixer.quit()
                self.mixer_initialized = False
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


class TextProcessor:
    """Text processing for clipboard text capture"""

    def get_selected_text(self) -> Optional[str]:
        """Get currently selected text"""
        try:
            selected_text = self._get_selected_text_native()
            if selected_text:
                return selected_text
            return self._get_selected_text_clipboard()
        except Exception as e:
            logger.error(f"Error getting selected text: {e}")
            return None

    def _get_selected_text_native(self) -> Optional[str]:
        """Get selected text via Accessibility API"""
        try:
            import Cocoa
            import Quartz
            from AppKit import NSWorkspace

            workspace = NSWorkspace.sharedWorkspace()
            active_app = workspace.frontmostApplication()

            if not active_app:
                return None

            pid = active_app.processIdentifier()
            app_ref = Quartz.AXUIElementCreateApplication(pid)

            if not app_ref:
                return None

            focused_result = Quartz.AXUIElementCopyAttributeValue(
                app_ref, Quartz.kAXFocusedUIElementAttribute, None
            )

            if focused_result[0] != 0 or not focused_result[1]:
                return None

            focused_element = focused_result[1]

            try:
                selected_text_result = Quartz.AXUIElementCopyAttributeValue(
                    focused_element, Quartz.kAXSelectedTextAttribute, None
                )

                if selected_text_result[0] == 0 and selected_text_result[1]:
                    selected_text = str(selected_text_result[1])
                    if selected_text and selected_text.strip():
                        return selected_text.strip()
            except Exception:
                pass

            return None

        except ImportError:
            return None
        except Exception:
            return None

    def _get_selected_text_clipboard(self) -> Optional[str]:
        """Fallback using clipboard"""
        try:
            original_clipboard = None
            try:
                original_clipboard = pyperclip.paste()
            except Exception:
                pass

            try:
                import Quartz

                cmd_down = Quartz.CGEventCreateKeyboardEvent(None, 0x37, True)
                c_down = Quartz.CGEventCreateKeyboardEvent(None, 0x08, True)
                c_up = Quartz.CGEventCreateKeyboardEvent(None, 0x08, False)
                cmd_up = Quartz.CGEventCreateKeyboardEvent(None, 0x37, False)

                Quartz.CGEventSetFlags(c_down, Quartz.kCGEventFlagMaskCommand)
                Quartz.CGEventSetFlags(c_up, Quartz.kCGEventFlagMaskCommand)

                Quartz.CGEventPost(Quartz.kCGHIDEventTap, cmd_down)
                Quartz.CGEventPost(Quartz.kCGHIDEventTap, c_down)
                Quartz.CGEventPost(Quartz.kCGHIDEventTap, c_up)
                Quartz.CGEventPost(Quartz.kCGHIDEventTap, cmd_up)

                time.sleep(0.05)

                selected_text = pyperclip.paste()

                if selected_text and selected_text.strip():
                    if original_clipboard and selected_text == original_clipboard:
                        return None

                    if original_clipboard and original_clipboard != selected_text:
                        time.sleep(0.05)
                        pyperclip.copy(original_clipboard)

                    return selected_text.strip()

                return None

            except ImportError:
                return None

        except Exception:
            return None

    def prepare_text(self, text: str) -> Optional[str]:
        """Clean and prepare text for TTS"""
        try:
            if not text:
                return None

            import re

            text = text.strip()

            def process_newlines(text):
                lines = text.split('\n')
                processed = []
                for i, line in enumerate(lines):
                    line = line.strip()
                    if line:
                        if i < len(lines) - 1 and line[-1] not in '.!?;:':
                            line += '.'
                        processed.append(line)
                return ' '.join(processed)

            text = process_newlines(text)
            text = re.sub(r'\s+', ' ', text)
            text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
            text = text.replace('\t', ' ')
            text = re.sub(r'\s+', ' ', text).strip()

            if not text or not any(c.isalnum() for c in text):
                return None

            return text

        except Exception as e:
            logger.error(f"Error preparing text: {e}")
            return None


class HotkeyMonitor:
    """Hotkey monitoring for double Option key"""

    def __init__(self, config: AppConfig):
        self.config = config
        self.listener = None
        self.last_cmd_press = 0
        self.cmd_pressed = False
        self.hotkey_callback = None
        self.stop_callback = None
        self.pause_callback = None
        self.running = False

    def start_monitoring(self) -> bool:
        """Start listening for hotkeys"""
        try:
            self.running = True

            self.listener = keyboard.Listener(
                on_press=self._on_key_press,
                on_release=self._on_key_release
            )

            self.listener.start()
            logger.info("Hotkey monitoring started")
            return True

        except Exception as e:
            logger.error(f"Failed to start hotkey monitoring: {e}")
            return False

    def stop_monitoring(self) -> None:
        """Stop listening"""
        try:
            self.running = False
            if self.listener:
                self.listener.stop()
                self.listener = None
            logger.info("Hotkey monitoring stopped")
        except Exception as e:
            logger.error(f"Error stopping hotkey: {e}")

    def on_hotkey_detected(self, callback: Callable) -> None:
        """Register hotkey callback"""
        self.hotkey_callback = callback

    def on_stop_requested(self, callback: Callable) -> None:
        """Register stop callback"""
        self.stop_callback = callback

    def on_pause_requested(self, callback: Callable) -> None:
        """Register pause callback"""
        self.pause_callback = callback

    def _on_key_press(self, key) -> None:
        """Handle key press"""
        try:
            if key == keyboard.Key.esc:
                logger.info("Escape key pressed - triggering stop")
                if self.stop_callback and self.running:
                    # Run in thread to avoid blocking keyboard listener
                    threading.Thread(target=self.stop_callback, daemon=True, name="StopHandler").start()
                return

            if key == keyboard.Key.alt or key == keyboard.Key.alt_r:
                current_time = time.time()

                if self.cmd_pressed and (current_time - self.last_cmd_press) <= self.config.hotkey_timeout:
                    logger.info("Double Option key detected!")

                    if self.pause_callback and self.running:
                        threading.Thread(target=self.pause_callback, daemon=True).start()
                    elif self.hotkey_callback and self.running:
                        threading.Thread(target=self.hotkey_callback, daemon=True).start()

                    self.cmd_pressed = False
                    self.last_cmd_press = 0
                else:
                    self.cmd_pressed = True
                    self.last_cmd_press = current_time

        except Exception as e:
            logger.error(f"Error in key press handler: {e}")

    def _on_key_release(self, key) -> None:
        """Handle key release"""
        try:
            if key != keyboard.Key.alt and key != keyboard.Key.alt_r:
                if self.cmd_pressed:
                    current_time = time.time()
                    if (current_time - self.last_cmd_press) > self.config.hotkey_timeout:
                        self.cmd_pressed = False
                        self.last_cmd_press = 0
        except Exception as e:
            logger.error(f"Error in key release handler: {e}")

    def cleanup(self) -> None:
        """Clean up"""
        self.stop_monitoring()


class OratorMenuBarApp(rumps.App):
    """Main menu bar application"""

    CONFIG_FILE = "config_hotkey.json"

    def __init__(self):
        # Get the app directory
        self.app_dir = os.path.dirname(os.path.abspath(__file__))

        # Load configuration
        self.config = self._load_config()

        # Initialize with icon (quit_button=None to add our own with cleanup)
        icon_path = os.path.join(self.app_dir, "orator_menu_icon.png")
        if os.path.exists(icon_path):
            super().__init__("Orator", icon=icon_path, quit_button=None)
        else:
            super().__init__("🔊", quit_button=None)

        # Engine state
        self.engine_running = False
        self.tts_engine: Optional[ONNXTTSEngine] = None
        self.audio_player: Optional[AudioPlayer] = None
        self.text_processor: Optional[TextProcessor] = None
        self.hotkey_monitor: Optional[HotkeyMonitor] = None

        # Build menu
        self._build_menu()

    def _load_config(self) -> AppConfig:
        """Load configuration from file"""
        config_path = os.path.join(self.app_dir, self.CONFIG_FILE)

        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    data = json.load(f)

                speed = data.get("speed", 1.0)
                if not isinstance(speed, (int, float)):
                    speed = 1.0
                speed = max(0.5, min(2.0, speed))

                return AppConfig(
                    model_path=os.path.join(self.app_dir, "kokoro_model_onnx/kokoro-v1_0.onnx"),
                    tokenizer_path=os.path.join(self.app_dir, "kokoro_model_onnx/tokenizer.json"),
                    voices_dir=os.path.join(self.app_dir, "kokoro_model_onnx/voices"),
                    voice=data.get("voice", "bf_isabella"),
                    hotkey_timeout=data.get("hotkey_timeout", 0.5),
                    max_text_length=data.get("max_text_length"),
                    device="cpu",
                    speed=speed
                )

            except Exception as e:
                logger.warning(f"Failed to load config: {e}")

        return AppConfig(
            model_path=os.path.join(self.app_dir, "kokoro_model_onnx/kokoro-v1_0.onnx"),
            tokenizer_path=os.path.join(self.app_dir, "kokoro_model_onnx/tokenizer.json"),
            voices_dir=os.path.join(self.app_dir, "kokoro_model_onnx/voices"),
        )

    def _save_config(self) -> None:
        """Save configuration to file"""
        config_path = os.path.join(self.app_dir, self.CONFIG_FILE)

        try:
            data = {
                "model_path": "kokoro_model_onnx/kokoro-v1_0.onnx",
                "tokenizer_path": "kokoro_model_onnx/tokenizer.json",
                "voices_dir": "kokoro_model_onnx/voices",
                "voice": self.config.voice,
                "hotkey_timeout": self.config.hotkey_timeout,
                "max_text_length": self.config.max_text_length,
                "device": "cpu",
                "log_level": "INFO",
                "speed": self.config.speed,
                "_comment": "Press Escape key to stop TTS playback at any time"
            }

            with open(config_path, 'w') as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save config: {e}")

    def _build_menu(self) -> None:
        """Build the menu structure"""
        # Status item
        self.status_item = rumps.MenuItem("Status: Stopped", callback=None)
        self.status_item.set_callback(None)

        # Current voice display item
        self.voice_display_item = rumps.MenuItem(f"Voice: {self.config.voice}", callback=None)
        self.voice_display_item.set_callback(None)

        # Start/Stop item
        self.start_stop_item = rumps.MenuItem("Start Engine", callback=self.toggle_engine)

        # Voice submenu
        self.voice_menu = self._build_voice_menu()

        # Speed submenu
        self.speed_menu = self._build_speed_menu()

        # Quit item (we use quit_button=None in __init__ to add our own with cleanup)
        quit_item = rumps.MenuItem("Quit", callback=self.quit_app)

        # Build menu
        self.menu = [
            self.status_item,
            self.voice_display_item,
            None,  # Separator
            self.start_stop_item,
            None,  # Separator
            ("Voices", self.voice_menu),
            ("Speed", self.speed_menu),
            None,  # Separator
            quit_item
        ]

    def _get_voice_language_group(self, voice_name: str) -> Optional[str]:
        """Get the language group name for a given voice"""
        for group_name, voices in VOICE_GROUPS.items():
            if voice_name in voices:
                return group_name
        return None

    def _build_voice_menu(self) -> list:
        """Build voice selection submenu grouped by language"""
        voice_items = []
        current_voice_group = self._get_voice_language_group(self.config.voice)

        for group_name, voices in VOICE_GROUPS.items():
            # Create submenu for this language group
            group_items = []
            group_has_current_voice = (group_name == current_voice_group)

            for voice in voices:
                item = rumps.MenuItem(
                    voice,
                    callback=lambda sender, v=voice: self.select_voice(v)
                )
                # Mark current voice
                if voice == self.config.voice:
                    item.state = 1
                group_items.append(item)

            # Create the group menu item with checkmark if it contains current voice
            group_menu_item = rumps.MenuItem(group_name)
            if group_has_current_voice:
                group_menu_item.state = 1

            voice_items.append((group_menu_item, group_items))

        return voice_items

    def _build_speed_menu(self) -> list:
        """Build speed selection submenu"""
        speeds = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
        speed_items = []

        for speed in speeds:
            item = rumps.MenuItem(
                f"{speed}x",
                callback=lambda sender, s=speed: self.select_speed(s)
            )
            if abs(speed - self.config.speed) < 0.01:
                item.state = 1
            speed_items.append(item)

        return speed_items

    def _update_voice_menu_state(self) -> None:
        """Update checkmarks in voice menu and voice display"""
        try:
            # Update voice display item
            self.voice_display_item.title = f"Voice: {self.config.voice}"

            # Get current voice's language group
            current_voice_group = self._get_voice_language_group(self.config.voice)

            for group_name in VOICE_GROUPS:
                if group_name in self.menu["Voices"]:
                    # Update group checkmark
                    group_item = self.menu["Voices"][group_name]
                    if hasattr(group_item, 'state'):
                        group_item.state = 1 if group_name == current_voice_group else 0

                    # Update individual voice checkmarks
                    for voice in VOICE_GROUPS[group_name]:
                        if voice in self.menu["Voices"][group_name]:
                            self.menu["Voices"][group_name][voice].state = 1 if voice == self.config.voice else 0
        except Exception as e:
            logger.error(f"Error updating voice menu: {e}")

    def _update_speed_menu_state(self) -> None:
        """Update checkmarks in speed menu"""
        try:
            speeds = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
            for speed in speeds:
                key = f"{speed}x"
                if key in self.menu["Speed"]:
                    self.menu["Speed"][key].state = 1 if abs(speed - self.config.speed) < 0.01 else 0
        except Exception as e:
            logger.error(f"Error updating speed menu: {e}")

    def toggle_engine(self, sender) -> None:
        """Start or stop the TTS engine"""
        if self.engine_running:
            self.stop_engine()
        else:
            self.start_engine()

    def start_engine(self) -> None:
        """Start the TTS engine"""
        try:
            self.status_item.title = "Status: Starting..."

            # Initialize TTS engine
            self.tts_engine = ONNXTTSEngine(self.config)
            if not self.tts_engine.initialize_model():
                self.status_item.title = "Status: Error"
                safe_notification("Orator", "Engine Error", "Failed to initialize TTS engine")
                return

            # Initialize audio player
            self.audio_player = AudioPlayer()
            if not self.audio_player.initialize():
                self.status_item.title = "Status: Error"
                return

            # Initialize text processor
            self.text_processor = TextProcessor()

            # Initialize hotkey monitor
            self.hotkey_monitor = HotkeyMonitor(self.config)
            self.hotkey_monitor.on_hotkey_detected(self.handle_hotkey)
            self.hotkey_monitor.on_stop_requested(self.handle_stop_request)
            self.hotkey_monitor.on_pause_requested(self.handle_pause_request)

            if not self.hotkey_monitor.start_monitoring():
                self.status_item.title = "Status: Error"
                return

            self.engine_running = True
            self.status_item.title = "Status: Running ✓"
            self.start_stop_item.title = "Stop Engine"

            safe_notification(
                "Orator",
                "Engine Started",
                f"Voice: {self.config.voice}\nDouble-tap Option key to speak"
            )

            logger.info("TTS engine started successfully")

        except Exception as e:
            logger.error(f"Failed to start engine: {e}")
            self.status_item.title = "Status: Error"
            safe_notification("Orator", "Error", str(e))

    def stop_engine(self) -> None:
        """Stop the TTS engine"""
        try:
            self.status_item.title = "Status: Stopping..."

            if self.hotkey_monitor:
                self.hotkey_monitor.cleanup()
                self.hotkey_monitor = None

            if self.audio_player:
                self.audio_player.cleanup()
                self.audio_player = None

            if self.tts_engine:
                self.tts_engine.cleanup()
                self.tts_engine = None

            self.text_processor = None

            self.engine_running = False
            self.status_item.title = "Status: Stopped"
            self.start_stop_item.title = "Start Engine"

            logger.info("TTS engine stopped")

        except Exception as e:
            logger.error(f"Error stopping engine: {e}")
            self.status_item.title = "Status: Error"

    def select_voice(self, voice: str) -> None:
        """Change the current voice"""
        try:
            old_voice = self.config.voice
            self.config.voice = voice

            if self.tts_engine and self.engine_running:
                if not self.tts_engine.set_voice(voice):
                    self.config.voice = old_voice
                    safe_notification("Orator", "Error", f"Failed to load voice: {voice}")
                    return

            self._update_voice_menu_state()
            self._save_config()

            safe_notification("Orator", "Voice Changed", f"Now using: {voice}")
            logger.info(f"Voice changed to: {voice}")

        except Exception as e:
            logger.error(f"Error changing voice: {e}")

    def select_speed(self, speed: float) -> None:
        """Change the TTS speed"""
        try:
            self.config.speed = speed

            if self.tts_engine:
                self.tts_engine.config.speed = speed

            self._update_speed_menu_state()
            self._save_config()

            logger.info(f"Speed changed to: {speed}x")

        except Exception as e:
            logger.error(f"Error changing speed: {e}")

    def handle_hotkey(self) -> None:
        """Handle hotkey trigger"""
        logger.info("Hotkey detected")

        if not self.engine_running or not self.tts_engine:
            return

        try:
            self.audio_player.play_notification("success")

            selected_text = self.text_processor.get_selected_text()

            if not selected_text:
                logger.info("No text selected")
                self.audio_player.play_notification("no_text")
                return

            prepared_text = self.text_processor.prepare_text(selected_text)

            if not prepared_text:
                self.audio_player.play_notification("error")
                return

            logger.info(f"Processing text: {len(prepared_text)} chars")

            audio_chunks = self.tts_engine.generate_audio_stream(prepared_text)
            self.audio_player.play_audio_stream(audio_chunks)

        except Exception as e:
            logger.error(f"Error in hotkey handler: {e}")
            if self.audio_player:
                self.audio_player.play_notification("error")

    def handle_stop_request(self) -> None:
        """Handle stop request (Escape key)"""
        logger.info("Stop request received - stopping audio playback")
        if self.audio_player:
            self.audio_player.stop_current()
            logger.info("Stop request processed")
        else:
            logger.warning("Audio player not available for stop request")

    def handle_pause_request(self) -> None:
        """Handle pause/resume or start TTS"""
        logger.info("Pause/Resume request")

        if not self.audio_player:
            return

        # If nothing playing, start TTS
        if not self.audio_player.is_active():
            self.audio_player.play_notification_overlay("success")

            if self.text_processor and self.tts_engine:
                selected_text = self.text_processor.get_selected_text()
                prepared_text = self.text_processor.prepare_text(selected_text or "") if selected_text else None

                if prepared_text:
                    audio_chunks = self.tts_engine.generate_audio_stream(prepared_text)
                    self.audio_player.play_audio_stream(audio_chunks)
                else:
                    self.audio_player.play_notification_overlay("no_text")
            return

        # Toggle pause/resume
        was_paused = self.audio_player.is_paused()
        self.audio_player.toggle_pause()
        now_paused = self.audio_player.is_paused()

        if now_paused and not was_paused:
            self.audio_player.play_notification_overlay("error")
        elif not now_paused and was_paused:
            self.audio_player.play_notification_overlay("success")

    def quit_app(self, sender) -> None:
        """Quit the application"""
        self.stop_engine()
        rumps.quit_application()


def main():
    """Application entry point"""
    logger.info("Starting Orator Menu Bar Application")

    # Create and run the app
    app = OratorMenuBarApp()

    # Auto-start engine on launch
    threading.Timer(1.0, app.start_engine).start()

    app.run()


if __name__ == "__main__":
    main()

