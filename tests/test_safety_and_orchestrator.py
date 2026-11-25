"""Automated tests for SafetyGuard confirmation, fast-path orchestrator, and kill-switch parsing."""

from __future__ import annotations

from pathlib import Path

from config.settings import load_config
from core.hotkey import parse_hotkey
from core.orchestrator import AgentOrchestrator, fast_path_route, strip_wake_prefix
from core.safety import SafetyGuard, assess_action_safety, is_explicit_confirmation
from tools.mac_tools import MacToolRegistry


def test_strip_wake_prefix():
    assert strip_wake_prefix("Jarvis, open VS Code") == "open VS Code"
    assert strip_wake_prefix("Hey Jarvis! pause the video") == "pause the video"
    assert strip_wake_prefix("open Safari") == "open Safari"


def test_fast_path_routing_coverage():
    assert fast_path_route("Jarvis, open VS Code") == [
        ("manage_application", {"action": "open", "app_name": "vs code"})
    ]
    assert fast_path_route("quit Safari") == [
        ("manage_application", {"action": "quit", "app_name": "safari"})
    ]
    assert fast_path_route("Search and play interstellar theme on YouTube") == [
        ("youtube_search_and_play", {"query": "interstellar theme"})
    ]
    assert fast_path_route("skip ad") == [("media_control", {"action": "skip_ad"})]
    assert fast_path_route("toggle fullscreen") == [("media_control", {"action": "fullscreen"})]
    assert fast_path_route("set volume to 45%") == [
        ("control_os_setting", {"setting": "volume", "value": "45"})
    ]
    assert fast_path_route("turn on dark mode") == [
        ("control_os_setting", {"setting": "dark_mode", "value": "on"})
    ]
    assert fast_path_route("what window is active") == [("get_active_window", {})]


def test_safety_assessment_destructive_categories():
    del_assess = assess_action_safety("delete_files", {"paths": ["/tmp/a.txt", "/tmp/b.txt"]})
    assert del_assess.requires_confirmation is True
    assert del_assess.category == "file_deletion"
    assert "delete 2 items" in del_assess.confirmation_prompt

    rm_assess = assess_action_safety("run_shell_command", {"command": "rm -rf ~/Downloads/old"})
    assert rm_assess.requires_confirmation is True
    assert rm_assess.category == "destructive_shell"

    safe_shell = assess_action_safety("run_shell_command", {"command": "ls -la ~/Desktop"})
    assert safe_shell.requires_confirmation is False

    mail_assess = assess_action_safety(
        "send_email", {"recipient": "alice@example.com", "subject": "Hi", "body": "Test"}
    )
    assert mail_assess.requires_confirmation is True
    assert "alice@example.com" in mail_assess.confirmation_prompt

    pwr_assess = assess_action_safety("system_power", {"action": "shutdown"})
    assert pwr_assess.requires_confirmation is True


def test_explicit_confirmation_phrase_matching():
    assert is_explicit_confirmation("Yes, execute.") is True
    assert is_explicit_confirmation("yes execute") is True
    assert is_explicit_confirmation("no cancel") is False
    assert is_explicit_confirmation("maybe later") is False
    assert is_explicit_confirmation("") is False


def test_safety_guard_blocks_or_allows_destructive_file_delete(tmp_path: Path):
    target = tmp_path / "secret.txt"
    target.write_text("important", encoding="utf-8")

    cfg = load_config()
    cfg.safety.audit_log_path = str(tmp_path / "audit.log")
    spoken_prompts: list[str] = []

    # 1. Negative / non-explicit confirmation cancels deletion
    guard_deny = SafetyGuard(
        config=cfg,
        speak_fn=spoken_prompts.append,
        listen_confirmation_fn=lambda _: "no stop",
    )
    tools_deny = MacToolRegistry(safety_guard=guard_deny)
    res_deny = tools_deny.execute("delete_files", {"paths": [str(target)], "permanent": True})
    assert res_deny.ok is False
    assert res_deny.data == {"blocked_by_safety": True}
    assert target.exists()  # File untouched!
    assert len(spoken_prompts) == 1

    # 2. Explicit "Yes, execute" authorizes deletion
    guard_allow = SafetyGuard(
        config=cfg,
        speak_fn=spoken_prompts.append,
        listen_confirmation_fn=lambda _: "Yes, execute",
    )
    tools_allow = MacToolRegistry(safety_guard=guard_allow)
    res_allow = tools_allow.execute("delete_files", {"paths": [str(target)], "permanent": True})
    assert res_allow.ok is True
    assert not target.exists()


def test_orchestrator_fast_path_executes_safe_tool(tmp_path: Path):
    cfg = load_config()
    cfg.safety.audit_log_path = str(tmp_path / "audit.log")
    tools = MacToolRegistry(safety_guard=SafetyGuard(config=cfg))
    orch = AgentOrchestrator(config=cfg, tools=tools)

    sample_file = tmp_path / "jarvis_test_doc.txt"
    sample_file.write_text("hello", encoding="utf-8")
    res = orch.handle_command(f"Jarvis, what window is active")
    assert res.route == "fast_path"
    assert len(res.tool_results) == 1


def test_parse_kill_switch_hotkey():
    mods, keycode = parse_hotkey("cmd+shift+escape")
    assert mods == {"cmd", "shift"}
    assert keycode == 53
