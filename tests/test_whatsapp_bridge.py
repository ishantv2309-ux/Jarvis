"""Automated tests for WhatsApp Desktop automation, ContactBook, and call controls."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from agent_tools import WebAgentEngine
from config.settings import load_config
from core.orchestrator import fast_path_route
from server import create_app
import whatsapp_bridge as wb


def test_contact_book_crud_and_resolution(tmp_path: Path):
    cpath = tmp_path / "contacts.json"
    book = wb.ContactBook(cpath)
    assert book.load() == {}

    book.upsert("Mom", "+91 98111-22333")
    book.upsert("Kunj", "+919988776655")
    assert len(book.load()) == 2

    m1 = book.resolve("mom")
    assert m1 is not None
    assert m1.name == "Mom"
    assert m1.phone_digits == "919811122333"

    m2 = book.resolve("unknown person")
    assert m2 is None

    m3 = book.resolve("+919876543210")
    assert m3 is not None
    assert m3.phone_digits == "919876543210"

    book.delete("Mom")
    assert "Mom" not in book.load()
    assert "Kunj" in book.load()


def test_whatsapp_fast_path_routing():
    assert fast_path_route("Jarvis, send a WhatsApp message to Mom saying I am reaching in 10 minutes") == [
        ("whatsapp_send_message", {"recipient": "Mom", "message": "I am reaching in 10 minutes"})
    ]
    assert fast_path_route("Message Kunj on WhatsApp saying check the repo") == [
        ("whatsapp_send_message", {"recipient": "Kunj", "message": "check the repo"})
    ]
    assert fast_path_route("Jarvis, call Rahul on WhatsApp") == [
        ("whatsapp_call", {"recipient": "Rahul", "call_type": "voice"})
    ]
    assert fast_path_route("Jarvis, video call Mom on WhatsApp") == [
        ("whatsapp_call", {"recipient": "Mom", "call_type": "video"})
    ]
    assert fast_path_route("answer the call") == [("whatsapp_answer_call", {})]
    assert fast_path_route("pick up") == [("whatsapp_answer_call", {})]
    assert fast_path_route("decline the call") == [("whatsapp_decline_call", {})]
    assert fast_path_route("reject") == [("whatsapp_decline_call", {})]


def test_whatsapp_send_and_call_url_vs_search_fallback(tmp_path: Path):
    cpath = tmp_path / "contacts.json"
    wb.ContactBook(cpath).upsert("Mom", "+919811122333")

    with patch("whatsapp_bridge.subprocess.run") as mock_run, patch(
        "whatsapp_bridge._run_osascript", return_value=(0, "", "")
    ) as mock_osa, patch("whatsapp_bridge.time.sleep"):
        mock_run.return_value.returncode = 0
        res_url = wb.whatsapp_send_message("Mom", "Hello Mom!", contacts_path=cpath)
        assert res_url["ok"] is True
        assert res_url["method"] == "phone_url"
        assert "Message sent to Mom, sir." == res_url["message"]
        opened_url = mock_run.call_args[0][0][1]
        assert opened_url.startswith("whatsapp://send?phone=919811122333&text=Hello%20Mom%21")
        assert mock_osa.called

    with patch("whatsapp_bridge._open_whatsapp_chat_by_search", return_value=True) as mock_search, patch(
        "whatsapp_bridge._run_osascript", return_value=(0, "", "")
    ), patch("whatsapp_bridge.time.sleep"):
        res_fb = wb.whatsapp_send_message("Rahul", "See you soon", contacts_path=cpath)
        assert res_fb["ok"] is True
        assert res_fb["method"] == "search_bar"
        mock_search.assert_called_once_with("Rahul")


def test_whatsapp_call_and_answer_signal_pause_mic(tmp_path: Path):
    cpath = tmp_path / "contacts.json"
    wb.ContactBook(cpath).upsert("Kunj", "+919988776655")

    with patch("whatsapp_bridge.subprocess.run"), patch(
        "whatsapp_bridge._run_osascript", return_value=(0, "", "")
    ), patch("whatsapp_bridge._click_ax_button_in_process", return_value=True), patch(
        "whatsapp_bridge.time.sleep"
    ):
        call_res = wb.whatsapp_call("Kunj", call_type="video", contacts_path=cpath)
        assert call_res["ok"] is True
        assert call_res["call_type"] == "video"
        assert call_res["pause_jarvis_mic"] is True

        ans_res = wb.whatsapp_answer_call()
        assert ans_res["ok"] is True
        assert ans_res["pause_jarvis_mic"] is True

        dec_res = wb.whatsapp_decline_call()
        assert dec_res["ok"] is True
        assert dec_res["pause_jarvis_mic"] is False


def test_agent_turn_propagates_pause_mic_and_contacts_api(tmp_path: Path):
    cfg = load_config()
    cfg.safety.audit_log_path = str(tmp_path / "audit.log")

    # Verify AgentTurnResult.pause_mic propagates from whatsapp_call
    engine = WebAgentEngine(config=cfg)
    with patch(
        "whatsapp_bridge.whatsapp_call",
        return_value={
            "ok": True,
            "pause_jarvis_mic": True,
            "message": "Calling Mom on WhatsApp, sir.",
        },
    ):
        turn = asyncio.run(engine.handle_user_input("Jarvis, call Mom on WhatsApp"))
        assert turn.pause_mic is True
        assert "Calling Mom on WhatsApp, sir." in turn.spoken_reply

    # Verify /api/contacts CRUD endpoints on FastAPI server
    app = create_app(config=cfg, warmup_models=False, enable_whatsapp_monitor=False)
    app.state.jarvis.contacts = wb.ContactBook(tmp_path / "contacts.json")

    with TestClient(app) as client:
        assert client.get("/api/contacts").json()["contacts"] == {}
        r1 = client.post("/api/contacts", json={"name": "Rahul", "phone": "+919800011222"})
        assert r1.status_code == 200
        assert r1.json()["contacts"]["Rahul"] == "+919800011222"
        r2 = client.delete("/api/contacts/Rahul")
        assert r2.status_code == 200
        assert r2.json()["contacts"] == {}
