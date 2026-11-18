"""One-key mode: a Gemini key alone does the reasoning and the transcription; SAM is skipped."""

from __future__ import annotations

import base64
import io
import json
import wave

import httpx
import pytest
import respx

from punchwalk.clients import asr, llm
from punchwalk.clients.base import PROVIDER_BASETEN
from punchwalk.config import get_settings, reset_settings


@pytest.fixture
def gemini_only(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("GEMINI_API_KEY", "gem-test")
    monkeypatch.setenv("MODEL_API_KEY", "")
    monkeypatch.setenv("BASETEN_API_KEY", "")
    monkeypatch.setenv("PUNCHWALK_CACHE", "readwrite")
    reset_settings()
    yield get_settings()
    reset_settings()


def _wav(seconds: float = 2.0, rate: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * int(seconds * rate))
    return buf.getvalue()


def test_flags_follow_the_keys(gemini_only) -> None:
    s = gemini_only
    assert s.use_gemini_reasoning and s.use_gemini_asr and not s.sam_available


def test_reasoning_routes_to_gemini(gemini_only) -> None:
    assert llm.model_candidates(PROVIDER_BASETEN) == [gemini_only.GEMINI_MODEL]
    client = llm._client(PROVIDER_BASETEN)
    assert "generativelanguage.googleapis.com" in str(client.base_url)
    assert client.api_key == "gem-test"


def test_baseten_key_wins_for_reasoning(monkeypatch: pytest.MonkeyPatch, gemini_only) -> None:
    monkeypatch.setenv("BASETEN_API_KEY", "bt")
    reset_settings()
    assert not get_settings().use_gemini_reasoning


def test_gemini_reply_is_reshaped_like_muse_voice(gemini_only) -> None:
    body = asr.gemini_body_to_meta_shape(
        {"turns": [{"start_s": 1.0, "end_s": 3.5, "text": " Kitchen, stove burner is burnt. "},
                   {"start_s": 4, "end_s": 99, "text": "Hallway outlet cover is cracked"},
                   {"start_s": 5, "end_s": 6, "text": "   "}]},
        audio_ms=8000,
    )
    turns = asr.parse_response(body).turns
    assert [t.text for t in turns] == ["Kitchen, stove burner is burnt.", "Hallway outlet cover is cracked"]
    assert (turns[0].start_ms, turns[0].end_ms) == (1000, 3500)
    assert turns[1].end_ms == 8000  # clamped to the audio length


@respx.mock
def test_transcribe_through_gemini_and_cache(gemini_only, tmp_path) -> None:
    wav = tmp_path / "audio.wav"
    wav.write_bytes(_wav())
    reply = {"candidates": [{"content": {"parts": [{"text": json.dumps(
        {"turns": [{"start_s": 0.2, "end_s": 1.5, "text": "Bedroom door does not latch"}]})}]}}]}
    route = respx.post(url__regex=r"https://generativelanguage\.googleapis\.com/v1beta/models/.+:generateContent").mock(
        return_value=httpx.Response(200, json=reply))

    first = asr.transcribe(wav, keywords=["strike plate"])
    assert [t.text for t in first.turns] == ["Bedroom door does not latch"]
    assert route.call_count == 1
    sent = route.calls[0].request
    assert sent.headers["x-goog-api-key"] == "gem-test"
    payload = json.loads(sent.content)
    inline = payload["contents"][0]["parts"][1]["inline_data"]
    assert inline["mime_type"] == "audio/wav" and base64.b64decode(inline["data"]).startswith(b"RIFF")
    assert "strike plate" in payload["contents"][0]["parts"][0]["text"]

    again = asr.transcribe(wav, keywords=["strike plate"])
    assert again.turns[0].text == first.turns[0].text
    assert route.call_count == 1  # served from the cache


@respx.mock
def test_gemini_error_is_an_api_error(gemini_only, tmp_path) -> None:
    wav = tmp_path / "audio.wav"
    wav.write_bytes(_wav())
    respx.post(url__regex=r".*generateContent").mock(return_value=httpx.Response(400, json={"error": "bad"}))
    with pytest.raises(asr.ApiError):
        asr.transcribe(wav)
