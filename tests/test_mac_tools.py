"""Tests for macOS tool execution, app resolution, file ops, and YouTube resolution."""

from __future__ import annotations

from pathlib import Path

from config.settings import load_config
from core.safety import SafetyGuard
from tools.mac_tools import MacToolRegistry, resolve_app_name


def test_resolve_app_aliases():
    assert resolve_app_name("VS Code") == "Visual Studio Code"
    assert resolve_app_name("chrome") == "Google Chrome"
    assert resolve_app_name("Finder") == "Finder"


def test_list_running_apps_and_active_window(tmp_path: Path):
    cfg = load_config()
    cfg.safety.audit_log_path = str(tmp_path / "audit.log")
    tools = MacToolRegistry(safety_guard=SafetyGuard(config=cfg))

    running = tools.execute("manage_application", {"action": "list_running"})
    assert running.ok is True
    assert isinstance(running.data["apps"], list)
    assert len(running.data["apps"]) > 0

    active = tools.execute("get_active_window", {})
    assert active.ok is True
    assert "app" in active.data


def test_file_management_organize_and_spotlight(tmp_path: Path):
    cfg = load_config()
    cfg.safety.audit_log_path = str(tmp_path / "audit.log")
    tools = MacToolRegistry(safety_guard=SafetyGuard(config=cfg))

    messy = tmp_path / "messy"
    messy.mkdir()
    (messy / "photo.png").write_bytes(b"png")
    (messy / "report.pdf").write_bytes(b"pdf")
    (messy / "song.mp3").write_bytes(b"mp3")

    res = tools.execute("manage_files", {"action": "organize_by_type", "source": str(messy)})
    assert res.ok is True
    assert res.data["moved_count"] == 3
    assert (messy / "Images" / "photo.png").exists()
    assert (messy / "Documents" / "report.pdf").exists()
    assert (messy / "Audio" / "song.mp3").exists()

    spot = tools.execute("search_files", {"query": "Finder", "directory": "/System/Library/CoreServices", "limit": 5})
    assert spot.ok is True


def test_youtube_url_resolution_shape():
    url, title = MacToolRegistry.resolve_youtube_top_url("lofi hip hop radio")
    assert url.startswith("https://www.youtube.com/")
    assert len(title) > 0
