"""Trust numbers, economics, audit export, footage purge and project erasure."""

from __future__ import annotations

import csv
import io
from pathlib import Path

from fastapi.testclient import TestClient
from sqlmodel import select

from punchwalk.api.routes.insights import _csv_safe, _sent_by_system_while_flagged
from punchwalk.core.enums import HistoryAction, HistoryActor, ItemStatus
from punchwalk.core.models import HistoryEvent, Item, Project, Walk, utcnow
from punchwalk.db import session_factory
from tests import test_api_support as support
from tests.test_api_items import add_items
from tests.test_api_support import ready_walk

api = support.api
clip_path = support.clip_path


def _event(actor: HistoryActor, action: HistoryAction, **kw) -> dict:
    return HistoryEvent(at=utcnow(), actor=actor, action=action, **kw).model_dump(mode="json", by_alias=True)


def _item_with(history: list[dict]) -> Item:
    return Item(project_id="p", number=1, origin_walk_id="w", title="t", history=history)


def test_invariant_flags_a_system_send_while_flagged() -> None:
    leak = _item_with([
        _event(HistoryActor.model, HistoryAction.flagged),
        _event(HistoryActor.system, HistoryAction.dispatched),
    ])
    assert _sent_by_system_while_flagged(leak)


def test_invariant_allows_send_after_human_approval() -> None:
    ok = _item_with([
        _event(HistoryActor.model, HistoryAction.flagged),
        _event(HistoryActor.human, HistoryAction.status_changed, field="status", from_="needs_review", to="open"),
        _event(HistoryActor.human, HistoryAction.dispatched),
    ])
    assert not _sent_by_system_while_flagged(ok)
    clean = _item_with([_event(HistoryActor.model, HistoryAction.verified), _event(HistoryActor.system, HistoryAction.dispatched)])
    assert not _sent_by_system_while_flagged(clean)


def test_csv_cells_cannot_be_formulas() -> None:
    assert _csv_safe("=HYPERLINK(\"http://x\")").startswith("'=")
    assert _csv_safe("+1") == "'+1"
    assert _csv_safe("-2") == "'-2"
    assert _csv_safe("@SUM(A1)") == "'@SUM(A1)"
    assert _csv_safe("Kitchen") == "Kitchen"
    assert _csv_safe(None) == ""


def test_trust_counts_what_a_person_decided(api: TestClient, clip_path: Path) -> None:
    project, _ = ready_walk(api, clip_path)
    pid = project["id"]
    with session_factory() as s:
        walk_id = s.exec(select(Walk.id).where(Walk.project_id == pid)).first()
    add_items(pid, walk_id)

    res = api.get(f"/api/projects/{pid}/trust")
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["items_total"] == 4
    assert body["held_for_a_person"] >= 1  # the needs_review item
    assert body["verified_clean"] == body["items_total"] - body["held_for_a_person"]
    assert body["flagged_sent_without_approval"] == 0
    assert body["outlines_shown"] + body["outlines_withheld"] == body["items_total"]
    assert 0 <= body["human_touch_rate"] <= 1


def test_trust_for_unknown_project_is_404(api: TestClient) -> None:
    assert api.get("/api/projects/prj_nope/trust").status_code == 404
    assert api.get("/api/projects/prj_nope/economics").status_code == 404
    assert api.get("/api/projects/prj_nope/audit.csv").status_code == 404


def test_economics_uses_stated_assumptions(api: TestClient, clip_path: Path) -> None:
    project, _ = ready_walk(api, clip_path)
    pid = project["id"]
    res = api.get(f"/api/projects/{pid}/economics")
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["items"] >= 1 and body["walks"] >= 1
    assert body["assumptions"]["labor_usd_per_hour"] == 65.0
    assert body["net_hours_saved"] <= body["manual_hours_replaced"]
    assert "estimate" in body["caveat"]

    cheaper = api.get(f"/api/projects/{pid}/economics", params={"manual_min_per_item": 0, "manual_min_per_walk": 0})
    assert cheaper.json()["net_hours_saved"] == 0


def test_audit_csv_lists_every_event_and_marks_dismissed(api: TestClient, clip_path: Path) -> None:
    project, _ = ready_walk(api, clip_path)
    pid = project["id"]
    with session_factory() as s:
        item = s.exec(select(Item).where(Item.project_id == pid)).first()
        item.title = "=cmd|' /C calc'!A0"
        s.add(item)
        s.commit()
        item_id = item.id
    api.delete(f"/api/items/{item_id}")

    res = api.get(f"/api/projects/{pid}/audit.csv")
    assert res.status_code == 200
    assert res.headers["content-type"].startswith("text/csv")
    assert "attachment" in res.headers["content-disposition"]
    rows = list(csv.reader(io.StringIO(res.text)))
    assert rows[0][:3] == ["item", "title", "dismissed"]
    body_rows = rows[1:]
    assert body_rows, "the dismissed item's history is still exported"
    assert all(r[1].startswith("'=") for r in body_rows)
    assert any(r[2] == "yes" and r[5] == "dismissed" for r in body_rows)


def test_purge_footage_keeps_evidence(api: TestClient, clip_path: Path) -> None:
    project, upload = ready_walk(api, clip_path)
    walk_id = upload["walk_id"]
    root = Path(support.get_settings().DATA_DIR) / "walks" / walk_id
    (root / "clips").mkdir(parents=True, exist_ok=True)
    (root / "clips" / "00_raw.mp4").write_bytes(b"x" * 1000)
    (root / "audio.wav").write_bytes(b"y" * 500)
    keep = root / "display.mp4"
    keep.write_bytes(keep.read_bytes() if keep.exists() else b"video")

    res = api.post(f"/api/walks/{walk_id}/purge-footage")
    assert res.status_code == 200, res.text
    out = res.json()
    assert out["bytes_freed"] >= 1500
    assert not (root / "clips").exists() and not (root / "audio.wav").exists()
    assert keep.exists()
    assert api.post("/api/walks/wlk_nope/purge-footage").status_code == 404


def test_erase_project_removes_rows_and_files(api: TestClient, clip_path: Path) -> None:
    project, upload = ready_walk(api, clip_path)
    pid = project["id"]
    walk_id = upload["walk_id"]
    root = Path(support.get_settings().DATA_DIR) / "walks" / walk_id
    assert root.exists()

    assert api.delete(f"/api/projects/{pid}").status_code == 204
    assert api.get(f"/api/projects/{pid}").status_code == 404
    assert not root.exists()
    with session_factory() as s:
        assert s.get(Project, pid) is None
        assert not s.exec(select(Item).where(Item.project_id == pid)).all()
        assert not s.exec(select(Walk).where(Walk.project_id == pid)).all()
    assert api.delete(f"/api/projects/{pid}").status_code == 404


def test_erase_is_blocked_in_demo_mode(api: TestClient, clip_path: Path, monkeypatch) -> None:
    project, _ = ready_walk(api, clip_path)
    monkeypatch.setenv("DEMO_MODE", "true")
    from punchwalk.config import reset_settings

    reset_settings()
    res = api.delete(f"/api/projects/{project['id']}")
    assert res.status_code == 403
    assert "demo" in res.json()["detail"].lower()
    assert ItemStatus.open  # keep the import honest


def test_readyz_reports_each_dependency(api: TestClient) -> None:
    res = api.get("/readyz")
    body = res.json()
    assert set(body["checks"]) == {"database", "ffmpeg", "data_dir_writable", "reasoning_key", "transcription_key"}
    assert body["checks"]["database"] is True
    assert res.status_code == (200 if body["ready"] else 503)
