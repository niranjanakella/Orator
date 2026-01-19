"""
CLI interface for Orator TTS
"""

import sys
import click
import logging
from .config_manager import ConfigManager, HotkeyConfig
from .daemon import DaemonManager
from .hotkey_recorder import HotkeyRecorder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# ASCII Art & Branding
# ═══════════════════════════════════════════════════════════════════════════════

ORATOR_BANNER_LARGE = """
   \033[1;38;5;183m ██████╗ ██████╗  █████╗ ████████╗ ██████╗ ██████╗ \033[0m
   \033[1;38;5;183m██╔═══██╗██╔══██╗██╔══██╗╚══██╔══╝██╔═══██╗██╔══██╗\033[0m
   \033[1;38;5;147m██║   ██║██████╔╝███████║   ██║   ██║   ██║██████╔╝\033[0m
   \033[1;38;5;147m██║   ██║██╔══██╗██╔══██║   ██║   ██║   ██║██╔══██╗\033[0m
   \033[1;38;5;111m╚██████╔╝██║  ██║██║  ██║   ██║   ╚██████╔╝██║  ██║\033[0m
   \033[1;38;5;111m ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝    ╚═════╝ ╚═╝  ╚═╝\033[0m

   \033[38;5;250m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\033[0m

   \033[3;38;5;183m"Why read when you can listen? Transform walls of text into\033[0m
   \033[3;38;5;183m smooth, natural speech — your personal narrator awaits."\033[0m

   \033[38;5;250m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\033[0m

   \033[38;5;245m✦ Select text  →  Trigger hotkey  →  Listen\033[0m

                              \033[38;5;245mBuilt with ♥ by \033[1;38;5;183mNiranjan Akella\033[0m
"""

def show_banner():
    """Display the banner"""
    click.echo(ORATOR_BANNER_LARGE)


def show_commands_help():
    """Display available commands with descriptions"""
    commands_info = [
        ("start", "Start Orator daemon with menu bar icon and TTS engine"),
        ("stop", "Stop Orator daemon and TTS engine"),
        ("status", "Check Orator daemon status"),
        ("restart", "Restart Orator daemon"),
        ("config trigger", "Record hotkey for trigger action"),
        ("config pause", "Record hotkey for pause action"),
        ("config stop", "Record hotkey for stop action"),
        ("config list", "List current hotkey configurations"),
        ("log", "Show last N lines of Orator log file"),
    ]
    
    click.echo()
    click.echo(f"\033[38;5;183m  ┌─────────────────────────────────────────────────────────────┐\033[0m")
    click.echo(f"\033[38;5;183m  │\033[0m  \033[1;38;5;147mAvailable Commands\033[0m                                      \033[38;5;183m│\033[0m")
    click.echo(f"\033[38;5;183m  └─────────────────────────────────────────────────────────────┘\033[0m")
    click.echo()
    
    for cmd, desc in commands_info:
        # Format command name with color
        cmd_formatted = f"\033[38;5;147m{cmd:<20}\033[0m"
        # Format description
        desc_formatted = f"\033[38;5;245m{desc}\033[0m"
        click.echo(f"    {cmd_formatted}  {desc_formatted}")
    
    click.echo()
    click.echo(f"  \033[38;5;245mFor more information on a command, use:\033[0m")
    click.echo(f"  \033[38;5;147m  orator <command> --help\033[0m")
    click.echo()


@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx):
    """Orator TTS - Text-to-speech with custom hotkeys"""
    # If no command provided, show elegant help
    if ctx.invoked_subcommand is None:
        show_banner()
        show_commands_help()


@cli.command()
def start():
    """Start Orator daemon with menu bar icon and TTS engine"""
    # Show the grand startup banner
    show_banner()
    
    daemon_manager = DaemonManager()
    
    if daemon_manager.is_running():
        pid = daemon_manager.get_pid()
        click.echo(f"\033[38;5;214m⚠  Orator is already running (PID: {pid})\033[0m")
        click.echo(f"\033[38;5;245m   Use '\033[38;5;183morator restart\033[38;5;245m' to restart or '\033[38;5;183morator stop\033[38;5;245m' to stop.\033[0m")
        sys.exit(1)
    
    click.echo(f"\033[38;5;147m▸ Initializing Orator daemon...\033[0m")
    
    try:
        # Import here to avoid issues if rumps is not available
        import subprocess
        import os
        
        # Get Python executable
        python_exe = sys.executable
        
        # Start daemon process
        process = subprocess.Popen(
            [python_exe, '-m', 'nj_orator.orator_app'],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True
        )
        
        # Save PID
        pid_file = daemon_manager.config_manager.get_pid_file_path()
        with open(pid_file, 'w') as f:
            f.write(str(process.pid))
        
        click.echo(f"\033[38;5;78m✓ Daemon started successfully\033[0m \033[38;5;245m(PID: {process.pid})\033[0m")
        click.echo(f"\033[38;5;78m✓ Menu bar icon loading...\033[0m")
        click.echo()
        click.echo(f"\033[38;5;183m  ┌─────────────────────────────────────────────────┐\033[0m")
        click.echo(f"\033[38;5;183m  │\033[0m  \033[38;5;245mSelect text → Press hotkey → Enjoy listening\033[0m  \033[38;5;183m│\033[0m")
        click.echo(f"\033[38;5;183m  └─────────────────────────────────────────────────┘\033[0m")
        click.echo()
        click.echo(f"\033[38;5;245m  Configure hotkeys: \033[38;5;147morator config trigger\033[0m")
        click.echo(f"\033[38;5;245m  View logs:         \033[38;5;147morator log\033[0m")
        click.echo()
        
    except Exception as e:
        logger.error(f"Failed to start daemon: {e}")
        click.echo(f"\033[38;5;196m✗ Error:\033[0m Failed to start daemon: {e}", err=True)
        sys.exit(1)


@cli.command()
def stop():
    """Stop Orator daemon and TTS engine"""
    daemon_manager = DaemonManager()
    
    if not daemon_manager.is_running():
        click.echo(f"\033[38;5;245mOrator is not running.\033[0m")
        sys.exit(1)
    
    pid = daemon_manager.get_pid()
    click.echo(f"\033[38;5;147m▸ Stopping Orator daemon\033[0m \033[38;5;245m(PID: {pid})...\033[0m")
    
    if daemon_manager.stop():
        click.echo(f"\033[38;5;78m✓ Orator daemon stopped successfully\033[0m")
    else:
        click.echo(f"\033[38;5;196m✗ Error:\033[0m Failed to stop daemon", err=True)
        sys.exit(1)


@cli.command()
def status():
    """Check Orator daemon status"""
    daemon_manager = DaemonManager()
    
    if daemon_manager.is_running():
        pid = daemon_manager.get_pid()
        click.echo(f"\033[38;5;78m● Orator is running\033[0m \033[38;5;245m(PID: {pid})\033[0m")
    else:
        click.echo(f"\033[38;5;245m○ Orator is not running\033[0m")


@cli.command()
def restart():
    """Restart Orator daemon"""
    daemon_manager = DaemonManager()
    
    if not daemon_manager.is_running():
        click.echo(f"\033[38;5;245mOrator is not running.\033[0m")
        click.echo(f"\033[38;5;147m▸ Starting daemon...\033[0m")
        if daemon_manager.start():
            click.echo(f"\033[38;5;78m✓ Orator daemon started successfully\033[0m")
        else:
            click.echo(f"\033[38;5;196m✗ Error:\033[0m Failed to start daemon", err=True)
            sys.exit(1)
    else:
        click.echo(f"\033[38;5;147m▸ Restarting Orator daemon...\033[0m")
        if daemon_manager.restart():
            click.echo(f"\033[38;5;78m✓ Orator daemon restarted successfully\033[0m")
        else:
            click.echo(f"\033[38;5;196m✗ Error:\033[0m Failed to restart daemon", err=True)
            sys.exit(1)


@cli.group()
def config():
    """Configure hotkeys"""
    pass


@config.command()
def trigger():
    """Record hotkey for trigger action"""
    click.echo("Recording hotkey for trigger action...")
    
    config_manager = ConfigManager()
    daemon_manager = DaemonManager(config_manager)
    recorder = HotkeyRecorder()
    
    hotkey_dict = recorder.record()
    
    if not hotkey_dict or not hotkey_dict.get("key"):
        click.echo("No hotkey recorded. Cancelled or timed out.")
        sys.exit(0)  # Exit with 0 (success) since user cancelled/timed out
    
    hotkey_config = HotkeyConfig.from_dict(hotkey_dict)
    
    if config_manager.set_hotkey("trigger", hotkey_config):
        display = recorder.format_hotkey_display(hotkey_dict)
        click.echo(f"Trigger hotkey set: {display}")
        
        # Auto-restart daemon if running
        if daemon_manager.is_running():
            click.echo("Restarting Orator daemon to apply changes...")
            if daemon_manager.restart():
                click.echo("Daemon restarted successfully")
            else:
                click.echo("Warning: Failed to restart daemon. Please restart manually.", err=True)
        else:
            click.echo("Note: Start Orator daemon for changes to take effect.")
    else:
        click.echo("Error: Failed to save hotkey configuration", err=True)
        sys.exit(1)


@config.command()
def pause():
    """Record hotkey for pause action"""
    click.echo("Recording hotkey for pause action...")
    
    config_manager = ConfigManager()
    daemon_manager = DaemonManager(config_manager)
    recorder = HotkeyRecorder()
    
    hotkey_dict = recorder.record()
    
    if not hotkey_dict or not hotkey_dict.get("key"):
        click.echo("No hotkey recorded. Cancelled or timed out.")
        sys.exit(0)  # Exit with 0 (success) since user cancelled/timed out
    
    hotkey_config = HotkeyConfig.from_dict(hotkey_dict)
    
    if config_manager.set_hotkey("pause", hotkey_config):
        display = recorder.format_hotkey_display(hotkey_dict)
        click.echo(f"Pause hotkey set: {display}")
        
        # Auto-restart daemon if running
        if daemon_manager.is_running():
            click.echo("Restarting Orator daemon to apply changes...")
            if daemon_manager.restart():
                click.echo("Daemon restarted successfully")
            else:
                click.echo("Warning: Failed to restart daemon. Please restart manually.", err=True)
        else:
            click.echo("Note: Start Orator daemon for changes to take effect.")
    else:
        click.echo("Error: Failed to save hotkey configuration", err=True)
        sys.exit(1)


@config.command(name='stop')
def config_stop():
    """Record hotkey for stop action"""
    click.echo("Recording hotkey for stop action...")
    
    config_manager = ConfigManager()
    daemon_manager = DaemonManager(config_manager)
    recorder = HotkeyRecorder()
    
    hotkey_dict = recorder.record()
    
    if not hotkey_dict or not hotkey_dict.get("key"):
        click.echo("No hotkey recorded. Cancelled or timed out.")
        sys.exit(0)  # Exit with 0 (success) since user cancelled/timed out
    
    hotkey_config = HotkeyConfig.from_dict(hotkey_dict)
    
    if config_manager.set_hotkey("stop", hotkey_config):
        display = recorder.format_hotkey_display(hotkey_dict)
        click.echo(f"Stop hotkey set: {display}")
        
        # Auto-restart daemon if running
        if daemon_manager.is_running():
            click.echo("Restarting Orator daemon to apply changes...")
            if daemon_manager.restart():
                click.echo("Daemon restarted successfully")
            else:
                click.echo("Warning: Failed to restart daemon. Please restart manually.", err=True)
        else:
            click.echo("Note: Start Orator daemon for changes to take effect.")
    else:
        click.echo("Error: Failed to save hotkey configuration", err=True)
        sys.exit(1)


@config.command()
def list():
    """List current hotkey configurations"""
    config_manager = ConfigManager()
    recorder = HotkeyRecorder()
    
    click.echo()
    click.echo(f"\033[38;5;183m  ┌─────────────────────────────────────────┐\033[0m")
    click.echo(f"\033[38;5;183m  │\033[0m      \033[1;38;5;147mHotkey Configuration\033[0m              \033[38;5;183m│\033[0m")
    click.echo(f"\033[38;5;183m  └─────────────────────────────────────────┘\033[0m")
    click.echo()
    
    for action in ["trigger", "pause", "stop"]:
        hotkey = config_manager.get_hotkey(action)
        icon = "🎯" if action == "trigger" else "⏸️ " if action == "pause" else "⏹️ "
        if hotkey:
            display = recorder.format_hotkey_display(hotkey.to_dict())
            click.echo(f"    {icon} \033[38;5;147m{action.capitalize():8}\033[0m  \033[38;5;78m{display}\033[0m")
        else:
            click.echo(f"    {icon} \033[38;5;147m{action.capitalize():8}\033[0m  \033[38;5;245mNot configured\033[0m")
    
    click.echo()
    click.echo(f"  \033[38;5;245mConfigure with:\033[0m \033[38;5;147morator config <trigger|pause|stop>\033[0m")
    click.echo()


@cli.command()
@click.option('--lines', '-n', default=100, help='Number of lines to show (default: 100)')
def log(lines):
    """Show last N lines of Orator log file"""
    import os
    from pathlib import Path
    
    log_file = Path.home() / ".orator" / "orator.log"
    
    if not log_file.exists():
        click.echo(f"\033[38;5;245mLog file not found: {log_file}\033[0m")
        click.echo(f"\033[38;5;245mLogs will be created when the daemon starts.\033[0m")
        sys.exit(1)
    
    try:
        # Read last N lines from log file
        with open(log_file, 'r', encoding='utf-8') as f:
            all_lines = f.readlines()
            
        if not all_lines:
            click.echo(f"\033[38;5;245mLog file is empty.\033[0m")
            return
        
        # Get last N lines
        last_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines
        
        click.echo()
        click.echo(f"\033[38;5;183m  📋 Orator Logs\033[0m \033[38;5;245m(last {len(last_lines)} lines)\033[0m")
        click.echo(f"\033[38;5;141m  {'─' * 76}\033[0m")
        
        for line in last_lines:
            # Color-code log levels
            line_stripped = line.rstrip()
            if " - ERROR - " in line_stripped:
                click.echo(f"  \033[38;5;196m{line_stripped}\033[0m")
            elif " - WARNING - " in line_stripped:
                click.echo(f"  \033[38;5;214m{line_stripped}\033[0m")
            elif " - INFO - " in line_stripped:
                click.echo(f"  \033[38;5;245m{line_stripped}\033[0m")
            else:
                click.echo(f"  {line_stripped}")
        
        click.echo(f"\033[38;5;141m  {'─' * 76}\033[0m")
        click.echo(f"  \033[38;5;245mLog file:\033[0m {log_file}")
        click.echo()
        
    except Exception as e:
        click.echo(f"\033[38;5;196m✗ Error:\033[0m reading log file: {e}", err=True)
        sys.exit(1)


def main():
    """Main entry point for CLI"""
    cli()


if __name__ == "__main__":
    main()

