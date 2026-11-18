"""Muse Voice Transcribe: `POST /v1/asr/transcribe` (multipart `request` JSON + `audio` WAV).

Inputs over 9 minutes or 30 MB are split into chunks; turn times are offset by each chunk's
start. 429 -> exponential backoff (max 5 tries); 500 -> retry once.
"""

from __future__ import annotations

import base64
import io
import json
import time
import wave
from pathlib import Path

from pydantic import BaseModel

from punchwalk.clients import calls, cost
from punchwalk.clients.base import (
    PROVIDER_META,
    ApiError,
    http_client,
    meta_headers,
    raise_for_status,
    sha256_bytes,
    with_retries,
    wrap_transport_errors,
)
from punchwalk.clients.cache import get_cache, make_key
from punchwalk.config import get_settings

ENDPOINT = "/asr/transcribe"
MAX_CHUNK_S = 9 * 60
MAX_CHUNK_BYTES = 30 * 1024 * 1024


class TranscriptTurn(BaseModel):
    turn_id: int
    start_ms: int
    end_ms: int
    text: str


class Transcript(BaseModel):
    transcript: str
    audio_duration_ms: int
    turns: list[TranscriptTurn]


def parse_response(body: dict, offset_ms: int = 0, first_turn_id: int = 1) -> Transcript:
    turns = []
    for i, t in enumerate(body.get("turns") or []):
        text = (t.get("transcript") or "").strip()
        if not text:
            continue
        turns.append(
            TranscriptTurn(
                turn_id=first_turn_id + i,
                start_ms=int(t.get("startMs", 0)) + offset_ms,
                end_ms=int(t.get("endMs", 0)) + offset_ms,
                text=text,
            )
        )
    # renumber densely after dropping empty turns
    for i, t in enumerate(turns):
        t.turn_id = first_turn_id + i
    return Transcript(
        transcript=(body.get("transcript") or "").strip(),
        audio_duration_ms=int(body.get("audioDurationMs") or 0),
        turns=turns,
    )


def split_wav(data: bytes, max_s: float | None = None, max_bytes: int | None = None) -> list[tuple[int, bytes]]:
    """Split WAV bytes into [(offset_ms, wav_bytes)], each within the duration and size limits."""
    max_s = MAX_CHUNK_S if max_s is None else max_s
    max_bytes = MAX_CHUNK_BYTES if max_bytes is None else max_bytes
    with wave.open(io.BytesIO(data), "rb") as w:
        params = w.getparams()
        rate, width, channels, n = w.getframerate(), w.getsampwidth(), w.getnchannels(), w.getnframes()
        frame_bytes = width * channels
        if n / rate <= max_s and len(data) <= max_bytes:
            return [(0, data)]
        per_chunk = min(int(max_s * rate), (max_bytes - 1024) // frame_bytes)
        chunks: list[tuple[int, bytes]] = []
        start = 0
        while start < n:
            frames = w.readframes(per_chunk)
            buf = io.BytesIO()
            with wave.open(buf, "wb") as out:
                out.setparams(params)
                out.writeframes(frames)
            chunks.append((int(start * 1000 / rate), buf.getvalue()))
            start += per_chunk
        return chunks


def _transcribe_chunk(wav: bytes, request: dict, desc: str) -> tuple[dict, bool]:
    s = get_settings()
    cache = get_cache()
    sha = sha256_bytes(wav)
    key = make_key(PROVIDER_META, ENDPOINT, request["model"], request, [sha])
    hit = cache.get(key)
    if hit is not None:
        calls.emit(
            calls.CallRecord(
                provider=PROVIDER_META, model=request["model"], endpoint=ENDPOINT, cached=True, cache_key=key,
                units=(hit.response.get("audioDurationMs") or 0) / 1000, unit_kind="audio_seconds",
            )
        )
        return hit.response, True
    cache.require_live(key, desc)
    cost.assert_under_cap()

    def call() -> dict:
        with wrap_transport_errors(PROVIDER_META, ENDPOINT, request["model"]), http_client() as client:
            resp = client.post(
                f"{s.META_API_BASE}{ENDPOINT}",
                headers=meta_headers(s),
                files={
                    "request": (None, json.dumps(request), "application/json"),
                    "audio": ("audio.wav", wav, "audio/wav"),
                },
            )
            raise_for_status(resp, PROVIDER_META, ENDPOINT, request["model"])
            return resp.json()

    def should_retry(err: ApiError, attempt: int) -> bool:
        if err.status == 429:
            return attempt < 5
        if err.status is not None and err.status >= 500:
            return attempt < 2
        return err.status is None and attempt < 2

    t0 = time.monotonic()
    record = calls.CallRecord(
        provider=PROVIDER_META, model=request["model"], endpoint=ENDPOINT, unit_kind="audio_seconds", cache_key=key
    )
    try:
        body = with_retries(call, should_retry=should_retry, backoff_s=lambda a: min(2.0**a, 30.0))
    except ApiError as err:
        record.ok, record.error = False, str(err)[:500]
        record.latency_ms = int((time.monotonic() - t0) * 1000)
        calls.emit(record)
        raise
    record.latency_ms = int((time.monotonic() - t0) * 1000)
    dur = int(body.get("audioDurationMs") or 0)
    record.units, record.est_cost_usd = dur / 1000, cost.asr_cost(dur)
    calls.emit(record)
    cache.put(key, provider=PROVIDER_META, endpoint=ENDPOINT, model=request["model"], description=desc, response=body)
    return body, False


GEMINI_ASR_PROMPT = (
    "Transcribe this audio of a construction inspector walking a unit and calling out defects. "
    "Split it into one turn per sentence. Give each turn's start and end time in seconds from the start of "
    "this audio (decimal numbers). Keep wording exactly as spoken. Likely terms: {keywords}. "
    "Skip silence. If there is no speech, return an empty list."
)
GEMINI_ASR_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "turns": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "start_s": {"type": "NUMBER"},
                    "end_s": {"type": "NUMBER"},
                    "text": {"type": "STRING"},
                },
                "required": ["start_s", "end_s", "text"],
            },
        }
    },
    "required": ["turns"],
}
GEMINI_MAX_CHUNK_S = 5 * 60  # keeps the inline audio well under Gemini's 20 MB request limit
GEMINI_MAX_CHUNK_BYTES = 12 * 1024 * 1024


def gemini_body_to_meta_shape(body: dict, audio_ms: int) -> dict:
    """Gemini's JSON turns (seconds) in the shape of a Muse Voice Transcribe response (ms), so parse_response is shared."""
    turns = []
    for t in body.get("turns") or []:
        text = str(t.get("text") or "").strip()
        if not text:
            continue
        start = max(0.0, float(t.get("start_s") or 0.0))
        end = max(start, float(t.get("end_s") or start))
        if audio_ms:
            end = min(end, audio_ms / 1000)
            start = min(start, end)
        turns.append({"transcript": text, "startMs": int(start * 1000), "endMs": int(end * 1000)})
    return {"transcript": " ".join(t["transcript"] for t in turns), "audioDurationMs": audio_ms, "turns": turns}


def _wav_ms(wav: bytes) -> int:
    with wave.open(io.BytesIO(wav), "rb") as w:
        return int(w.getnframes() * 1000 / w.getframerate())


def _transcribe_chunk_gemini(wav: bytes, request: dict, desc: str) -> tuple[dict, bool]:
    """One chunk through Gemini (generateContent, inline audio). Cached like every other model call."""
    s = get_settings()
    cache = get_cache()
    endpoint = f"/models/{s.GEMINI_MODEL}:generateContent"
    sha = sha256_bytes(wav)
    key = make_key(PROVIDER_META, endpoint, s.GEMINI_MODEL, request, [sha])
    audio_ms = _wav_ms(wav)
    hit = cache.get(key)
    if hit is not None:
        calls.emit(calls.CallRecord(provider=PROVIDER_META, model=s.GEMINI_MODEL, endpoint=endpoint, cached=True,
                                    cache_key=key, units=audio_ms / 1000, unit_kind="audio_seconds"))
        return hit.response, True
    cache.require_live(key, desc)
    cost.assert_under_cap()
    payload = {
        "contents": [{"parts": [
            {"text": GEMINI_ASR_PROMPT.format(keywords=", ".join(request.get("keywords") or []) or "none")},
            {"inline_data": {"mime_type": "audio/wav", "data": base64.b64encode(wav).decode()}},
        ]}],
        "generationConfig": {"responseMimeType": "application/json", "responseSchema": GEMINI_ASR_SCHEMA,
                             "temperature": 0},
    }

    def call() -> dict:
        with wrap_transport_errors(PROVIDER_META, endpoint, s.GEMINI_MODEL), http_client() as client:
            resp = client.post(f"{s.GEMINI_NATIVE_BASE}{endpoint}", json=payload,
                               headers={"x-goog-api-key": s.GEMINI_API_KEY.get_secret_value()})
            raise_for_status(resp, PROVIDER_META, endpoint, s.GEMINI_MODEL)
            return resp.json()

    def should_retry(err: ApiError, attempt: int) -> bool:
        return (err.status == 429 and attempt < 5) or (err.transient and attempt < 2)

    t0 = time.monotonic()
    record = calls.CallRecord(provider=PROVIDER_META, model=s.GEMINI_MODEL, endpoint=endpoint,
                              unit_kind="audio_seconds", cache_key=key)
    try:
        raw = with_retries(call, should_retry=should_retry, backoff_s=lambda a: min(2.0**a, 30.0))
        text = raw["candidates"][0]["content"]["parts"][0]["text"]
        body = gemini_body_to_meta_shape(json.loads(text), audio_ms)
    except ApiError as err:
        record.ok, record.error = False, str(err)[:500]
        record.latency_ms = int((time.monotonic() - t0) * 1000)
        calls.emit(record)
        raise
    except (KeyError, IndexError, ValueError, TypeError) as err:
        record.ok, record.error = False, f"unreadable Gemini reply: {err!s}"[:500]
        record.latency_ms = int((time.monotonic() - t0) * 1000)
        calls.emit(record)
        raise ApiError(PROVIDER_META, endpoint, None, record.error, model=s.GEMINI_MODEL) from err
    record.latency_ms = int((time.monotonic() - t0) * 1000)
    record.units, record.est_cost_usd = audio_ms / 1000, 0.0  # price not tracked for Gemini audio
    calls.emit(record)
    cache.put(key, provider=PROVIDER_META, endpoint=endpoint, model=s.GEMINI_MODEL, description=desc, response=body)
    return body, False


def transcribe(
    wav_path: str | Path,
    keywords: list[str] | None = None,
    language_bias: list[str] | None = None,
    *,
    mode: str = "ENDPOINTING",
) -> Transcript:
    """Transcribe a mono 16-bit PCM WAV (16 or 24 kHz). Returns turns with absolute times."""
    s = get_settings()
    data = Path(wav_path).read_bytes()
    request = {
        "mode": mode,
        "model": s.ASR_MODEL,
        "audioEncoding": "WAV",
        "languageBias": list(language_bias or ["English"]),
        "keywords": list(keywords or []),
    }
    texts: list[str] = []
    turns: list[TranscriptTurn] = []
    total_ms = 0
    gemini = s.use_gemini_asr
    chunks = split_wav(data, GEMINI_MAX_CHUNK_S, GEMINI_MAX_CHUNK_BYTES) if gemini else split_wav(data)
    for offset_ms, chunk in chunks:
        desc = f"ASR {Path(wav_path).name} @ {offset_ms} ms ({len(chunk)} bytes)"
        body, _ = (_transcribe_chunk_gemini if gemini else _transcribe_chunk)(chunk, request, desc)
        part = parse_response(body, offset_ms=offset_ms, first_turn_id=len(turns) + 1)
        texts.append(part.transcript)
        turns.extend(part.turns)
        total_ms = max(total_ms, offset_ms + part.audio_duration_ms)
    return Transcript(transcript=" ".join(t for t in texts if t), audio_duration_ms=total_ms, turns=turns)
