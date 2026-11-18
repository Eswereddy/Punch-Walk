"""Trust, economics and privacy: what a buyer, an auditor or a judge asks after "does it work?".

- GET    /api/projects/{id}/trust       human-oversight numbers: what the AI did alone, what a person decided
- GET    /api/projects/{id}/economics   what a walk costs vs. the hours it replaces (assumptions are explicit)
- GET    /api/projects/{id}/audit.csv   every logged event on every item, including dismissed ones
- POST   /api/walks/{id}/purge-footage  data minimisation: delete the raw clips and audio, keep the evidence
- DELETE /api/projects/{id}             right to erasure: every row and every file of the project

Everything here is derived from rows the pipeline already writes (item history, verification, closure,
the model-call ledger), so the numbers can't drift from what really happened.
"""

from __future__ import annotations

import csv
import io
import shutil
from typing import Any

from fastapi import APIRouter, Depends, Query, Response
from pydantic import BaseModel
from sqlmodel import Session, col, delete, select

from punchwalk.api.errors import ApiError, not_found
from punchwalk.config import get_settings
from punchwalk.core.enums import (
    ClosureDecision,
    HistoryAction,
    HistoryActor,
    ItemStatus,
    format_item_number,
)
from punchwalk.core.models import (
    Dispatch,
    Item,
    Job,
    ModelCall,
    Point,
    Project,
    Turn,
    Walk,
)
from punchwalk.db import get_session
from punchwalk.pipeline.context import CLIPS_DIR, walk_dir

router = APIRouter(prefix="/api", tags=["trust"])

# Stated assumptions for the economics view. They are inputs, not facts: the caller can override them,
# and the response repeats them so a reader never has to guess what the savings number rests on.
DEFAULT_MANUAL_MIN_PER_ITEM = 6.0  # sort photo, retype defect, find the trade, paste into a sheet
DEFAULT_MANUAL_MIN_PER_WALK_OVERHEAD = 20.0  # set up the sheet, group by trade, send, chase
DEFAULT_LABOR_USD_PER_HOUR = 65.0  # loaded cost of a project manager / superintendent
REVIEW_MIN_PER_FLAGGED_ITEM = 1.0  # a person still looks at every flagged item


class TrustOut(BaseModel):
    project_id: str
    items_total: int
    items_dismissed: int
    verified_clean: int  # passed the model's own check, no flag
    held_for_a_person: int  # flagged: needs_review at some point
    human_edited: int  # a person changed a field
    human_signed_off: int  # a person set the item to signed off
    auto_signed_off: int  # signed off by the re-walk check alone (no human click)
    false_fixed_claims_caught: int  # "marked fixed" but the re-walk photo disagreed
    flagged_sent_without_approval: int  # must be 0: the safety invariant
    outlines_shown: int
    outlines_withheld: int  # no outline beat a wrong outline
    human_touch_rate: float | None  # share of items a person edited, approved or signed off
    notes: list[str]


class EconomicsOut(BaseModel):
    project_id: str
    walks: int
    items: int
    model_cost_usd: float  # from the model-call ledger, all calls
    live_model_cost_usd: float  # only calls that were not served from the cache
    cost_per_item_usd: float | None
    manual_hours_replaced: float
    review_hours_added: float
    net_hours_saved: float
    labor_value_saved_usd: float
    return_multiple: float | None  # labor value saved / model cost; None when there is no cost to divide by
    assumptions: dict[str, float]
    caveat: str


def _project(session: Session, project_id: str) -> Project:
    project = session.get(Project, project_id)
    if project is None:
        raise not_found("This project")
    return project


def _events(item: Item) -> list[dict[str, Any]]:
    return [e for e in (item.history or []) if isinstance(e, dict)]


def _human(event: dict[str, Any], action: HistoryAction) -> bool:
    return event.get("actor") == HistoryActor.human.value and event.get("action") == action.value


def _closed_by_human(item: Item) -> bool:
    return any(
        _human(e, HistoryAction.status_changed) and e.get("to") == ItemStatus.closed.value for e in _events(item)
    )


def _was_flagged(item: Item) -> bool:
    if item.status == ItemStatus.needs_review:
        return True
    return any(e.get("action") == HistoryAction.flagged.value for e in _events(item))


def _sent_by_system_while_flagged(item: Item) -> bool:
    """The safety invariant, replayed from the item's own history: a flag stays on until a person
    moves the item out of needs_review; a system dispatch while it is on is a leak."""
    flagged = False
    for e in _events(item):
        action, actor = e.get("action"), e.get("actor")
        if action == HistoryAction.flagged.value:
            flagged = True
        elif actor == HistoryActor.human.value and e.get("field") == "status" and e.get("from") == "needs_review":
            flagged = False
        elif action == HistoryAction.dispatched.value and actor == HistoryActor.system.value and flagged:
            return True
    return False


def trust_numbers(session: Session, project_id: str) -> TrustOut:
    rows = session.exec(select(Item).where(Item.project_id == project_id)).all()
    live = [i for i in rows if i.dismissed_at is None]

    held = [i for i in live if _was_flagged(i)]
    edited = [i for i in live if any(_human(e, HistoryAction.edited) for e in _events(i))]
    human_closed = [i for i in live if _closed_by_human(i)]
    auto_closed = [
        i
        for i in live
        if i.status == ItemStatus.closed
        and not _closed_by_human(i)
        and (i.closure or {}).get("decision") == ClosureDecision.signed_off.value
    ]
    false_fixed = [
        i
        for i in live
        if (i.closure or {}).get("decision") == ClosureDecision.needs_review.value
        and (i.closure or {}).get("voice_status") == "fixed"
    ]
    leaked = [i for i in live if _sent_by_system_while_flagged(i)]
    with_outline = [i for i in live if ((i.evidence or {}).get("grounding") in ("sam_concept", "sam_box", "spark_box"))]
    touched = {i.id for i in edited} | {i.id for i in human_closed} | {i.id for i in held}

    notes = [
        "Items the model flagged are never sent until a person taps approve.",
        "No outline is drawn unless it passes a score and a second check; withheld outlines are counted, not hidden.",
    ]
    if leaked:
        notes.append(f"WARNING: {len(leaked)} flagged item(s) were sent without a human approval event.")

    return TrustOut(
        project_id=project_id,
        items_total=len(live),
        items_dismissed=len(rows) - len(live),
        verified_clean=len(live) - len(held),
        held_for_a_person=len(held),
        human_edited=len(edited),
        human_signed_off=len(human_closed),
        auto_signed_off=len(auto_closed),
        false_fixed_claims_caught=len(false_fixed),
        flagged_sent_without_approval=len(leaked),
        outlines_shown=len(with_outline),
        outlines_withheld=len(live) - len(with_outline),
        human_touch_rate=round(len(touched) / len(live), 3) if live else None,
        notes=notes,
    )


@router.get("/projects/{project_id}/trust", response_model=TrustOut)
def project_trust(project_id: str, session: Session = Depends(get_session)) -> TrustOut:
    _project(session, project_id)
    return trust_numbers(session, project_id)


@router.get("/projects/{project_id}/economics", response_model=EconomicsOut)
def project_economics(
    project_id: str,
    manual_min_per_item: float = Query(DEFAULT_MANUAL_MIN_PER_ITEM, ge=0, le=120),
    manual_min_per_walk: float = Query(DEFAULT_MANUAL_MIN_PER_WALK_OVERHEAD, ge=0, le=600),
    labor_usd_per_hour: float = Query(DEFAULT_LABOR_USD_PER_HOUR, ge=0, le=1000),
    session: Session = Depends(get_session),
) -> EconomicsOut:
    _project(session, project_id)
    walks = session.exec(select(Walk).where(Walk.project_id == project_id)).all()
    walk_ids = [w.id for w in walks]
    n_items = len(
        session.exec(select(Item).where(Item.project_id == project_id, col(Item.dismissed_at).is_(None))).all()
    )
    calls: list[ModelCall] = []
    if walk_ids:
        calls = list(session.exec(select(ModelCall).where(col(ModelCall.walk_id).in_(walk_ids))).all())
    total = sum(c.est_cost_usd or 0.0 for c in calls)
    live = sum(c.est_cost_usd or 0.0 for c in calls if not c.cached)

    initial_walks = [w for w in walks if str(getattr(w.kind, "value", w.kind)) == "initial"]
    manual_h = (n_items * manual_min_per_item + len(initial_walks) * manual_min_per_walk) / 60.0
    review_h = _flagged_count(session, project_id) * REVIEW_MIN_PER_FLAGGED_ITEM / 60.0
    net_h = max(manual_h - review_h, 0.0)
    value = net_h * labor_usd_per_hour

    return EconomicsOut(
        project_id=project_id,
        walks=len(walks),
        items=n_items,
        model_cost_usd=round(total, 4),
        live_model_cost_usd=round(live, 4),
        cost_per_item_usd=round(total / n_items, 4) if n_items else None,
        manual_hours_replaced=round(manual_h, 2),
        review_hours_added=round(review_h, 2),
        net_hours_saved=round(net_h, 2),
        labor_value_saved_usd=round(value, 2),
        return_multiple=round(value / total, 1) if total > 0 else None,
        assumptions={
            "manual_min_per_item": manual_min_per_item,
            "manual_min_per_walk_overhead": manual_min_per_walk,
            "labor_usd_per_hour": labor_usd_per_hour,
            "review_min_per_flagged_item": REVIEW_MIN_PER_FLAGGED_ITEM,
        },
        caveat=(
            "Model cost is measured from the call ledger. Hours saved is an estimate built from the stated "
            "assumptions, not a measurement; replace them with your own numbers via the query parameters."
        ),
    )


def _flagged_count(session: Session, project_id: str) -> int:
    rows = session.exec(
        select(Item).where(Item.project_id == project_id, col(Item.dismissed_at).is_(None))
    ).all()
    return sum(1 for i in rows if _was_flagged(i))


def _csv_safe(value: object) -> str:
    """Neutralise spreadsheet formula injection: a cell that starts with = + - @ is read as a formula."""
    text = "" if value is None else str(value)
    return "'" + text if text[:1] in ("=", "+", "-", "@", "\t", "\r") else text


@router.get("/projects/{project_id}/audit.csv", include_in_schema=False)
def project_audit_csv(project_id: str, session: Session = Depends(get_session)) -> Response:
    project = _project(session, project_id)
    items = session.exec(select(Item).where(Item.project_id == project_id).order_by(col(Item.number))).all()
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(["item", "title", "dismissed", "at", "actor", "action", "field", "from", "to", "note"])
    for item in items:
        for e in _events(item):
            writer.writerow(
                [
                    _csv_safe(format_item_number(item.number)),
                    _csv_safe(item.title),
                    "yes" if item.dismissed_at else "no",
                    _csv_safe(e.get("at")),
                    _csv_safe(e.get("actor")),
                    _csv_safe(e.get("action")),
                    _csv_safe(e.get("field")),
                    _csv_safe(e.get("from")),
                    _csv_safe(e.get("to")),
                    _csv_safe(e.get("note")),
                ]
            )
    slug = "".join(ch if ch.isalnum() else "-" for ch in project.name.lower()).strip("-") or "project"
    return Response(
        content=out.getvalue(),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{slug}-audit.csv"'},
    )


def _forbid_in_demo() -> None:
    if get_settings().DEMO_MODE:
        raise ApiError(403, "Deleting data is turned off in the public demo. It resets every night anyway.")


@router.post("/walks/{walk_id}/purge-footage")
def purge_footage(walk_id: str, session: Session = Depends(get_session)) -> dict[str, Any]:
    """Delete the raw uploaded clips and the extracted audio. The evidence photos, the display video the
    punch list plays, and every item stay. Bystanders' voices and any footage never used as evidence go."""
    _forbid_in_demo()
    walk = session.get(Walk, walk_id)
    if walk is None:
        raise not_found("This walk")
    root = walk_dir(walk_id)
    freed = 0
    removed: list[str] = []
    clips = root / CLIPS_DIR
    if clips.exists():
        freed += sum(p.stat().st_size for p in clips.rglob("*") if p.is_file())
        shutil.rmtree(clips, ignore_errors=True)
        removed.append(CLIPS_DIR)
    for name in ("audio.wav",):
        f = root / name
        if f.exists():
            freed += f.stat().st_size
            f.unlink(missing_ok=True)
            removed.append(name)
    return {"walk_id": walk_id, "removed": removed, "bytes_freed": freed}


@router.delete("/projects/{project_id}", status_code=204)
def erase_project(project_id: str, session: Session = Depends(get_session)) -> Response:
    """Erase a project: items, walks, transcripts, dispatch records, model-call rows and every file."""
    _forbid_in_demo()
    project = _project(session, project_id)
    walk_ids = [w.id for w in session.exec(select(Walk).where(Walk.project_id == project_id)).all()]
    # Children first: Item.origin_walk_id and Walk.rewalk_of are foreign keys.
    session.exec(delete(Item).where(Item.project_id == project_id))
    session.exec(delete(Dispatch).where(Dispatch.project_id == project_id))
    if walk_ids:
        session.exec(delete(Turn).where(col(Turn.walk_id).in_(walk_ids)))
        session.exec(delete(Point).where(col(Point.walk_id).in_(walk_ids)))
        session.exec(delete(ModelCall).where(col(ModelCall.walk_id).in_(walk_ids)))
        session.exec(delete(Job).where(col(Job.walk_id).in_(walk_ids)))
        session.exec(delete(Walk).where(col(Walk.rewalk_of).in_(walk_ids)))
        session.exec(delete(Walk).where(col(Walk.id).in_(walk_ids)))
    session.delete(project)
    session.commit()
    for wid in walk_ids:
        shutil.rmtree(walk_dir(wid), ignore_errors=True)
    return Response(status_code=204)
