"""
Daemon service management for Orator TTS
Handles background process management with PID file
"""

import os
import sys
import signal
import subprocess
import logging
from pathlib import Path
from typing import Optional

from .config_manager import ConfigManager

logger = logging.getLogger(__name__)


class DaemonManager:
    """Manages Orator daemon process"""

    def __init__(self, config_manager: Optional[ConfigManager] = None):
        """
        Initialize daemon manager
        
        Args:
            config_manager: ConfigManager instance. If None, creates new one.
        """
        self.config_manager = config_manager or ConfigManager()
        self.pid_file = self.config_manager.get_pid_file_path()

    def is_running(self) -> bool:
        """Check if daemon is currently running"""
        if not self.pid_file.exists():
            return False

        try:
            with open(self.pid_file, 'r') as f:
                pid = int(f.read().strip())

            # Check if process exists
            try:
                os.kill(pid, 0)  # Signal 0 doesn't kill, just checks existence
                return True
            except OSError:
                # Process doesn't exist, clean up stale PID file
                self.pid_file.unlink()
                return False
        except (ValueError, FileNotFoundError):
            return False

    def get_pid(self) -> Optional[int]:
        """Get daemon PID if running"""
        if not self.is_running():
            return None

        try:
            with open(self.pid_file, 'r') as f:
                return int(f.read().strip())
        except (ValueError, FileNotFoundError):
            return None

    def start(self) -> bool:
        """
        Start the daemon process
        
        Returns:
            True if started successfully, False otherwise
        """
        if self.is_running():
            pid = self.get_pid()
            logger.warning(f"Daemon is already running (PID: {pid})")
            return False

        try:
            # Get the path to the current Python interpreter
            python_exe = sys.executable
            
            # Get the path to the orator_app module
            import nj_orator.orator_app
            app_module_path = nj_orator.orator_app.__file__
            
            # Start daemon as background process
            process = subprocess.Popen(
                [python_exe, '-m', 'nj_orator.orator_app'],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True
            )
            pid = process.pid

            # Save PID to file
            with open(self.pid_file, 'w') as f:
                f.write(str(pid))

            logger.info(f"Daemon started with PID: {pid}")
            return True

        except Exception as e:
            logger.error(f"Failed to start daemon: {e}")
            return False

    def stop(self) -> bool:
        """
        Stop the daemon process
        
        Returns:
            True if stopped successfully, False otherwise
        """
        if not self.is_running():
            logger.warning("Daemon is not running")
            return False

        try:
            pid = self.get_pid()
            if pid is None:
                return False

            # Send SIGTERM for graceful shutdown
            os.kill(pid, signal.SIGTERM)

            # Wait a bit for graceful shutdown
            import time
            for _ in range(10):  # Wait up to 1 second
                time.sleep(0.1)
                try:
                    os.kill(pid, 0)  # Check if still exists
                except OSError:
                    # Process terminated
                    break
            else:
                # Force kill if still running
                try:
                    os.kill(pid, signal.SIGKILL)
                except OSError:
                    pass

            # Clean up PID file
            if self.pid_file.exists():
                self.pid_file.unlink()

            logger.info(f"Daemon stopped (PID: {pid})")
            return True

        except Exception as e:
            logger.error(f"Failed to stop daemon: {e}")
            return False

    def restart(self) -> bool:
        """
        Restart the daemon process
        
        Returns:
            True if restarted successfully, False otherwise
        """
        was_running = self.is_running()
        
        if was_running:
            logger.info("Restarting daemon...")
            if not self.stop():
                logger.error("Failed to stop daemon during restart")
                return False
            
            # Wait a moment for cleanup
            import time
            time.sleep(0.5)
        else:
            logger.info("Daemon not running, starting...")
        
        # Start the daemon
        return self.start()

    def cleanup_pid_file(self) -> None:
        """Clean up PID file (called on exit)"""
        if self.pid_file.exists():
            try:
                self.pid_file.unlink()
            except Exception as e:
                logger.error(f"Failed to cleanup PID file: {e}")

