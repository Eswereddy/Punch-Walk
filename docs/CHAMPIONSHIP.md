# Punchwalk: Championship evaluation guide

One page per judging criterion: the claim, the evidence in this repo, and what is still unproven.
Numbers come from `eval/report.md`, the cost ledger and the code. Anything that is a hypothesis is labelled as one.

## 1. Real value and problem validation (20%)

**Problem.** A final punch walk takes minutes. The write-up (sort photos, retype defects, split by trade,
send, chase) takes hours, and the contractor's final payment waits on it.

**What the product does about it.** The inspector talks while walking. Every item arrives with the exact
second of video, a photo, the inspector's own words, the trade and a severity. On the re-walk it signs off
what is fixed, with before/after photos, and catches "marked fixed, but still visible".

**Evidence in the repo**
- Three real walks on Ray-Ban Meta glasses, labelled by the inspector: 14/14 defects found, right trade and
  right room on all 14. Closure accuracy 11/12, both false "fixed" claims caught (`eval/report.md`).
- `GET /api/projects/{id}/economics` computes model cost from the ledger and hours saved from stated,
  overridable assumptions (6 min per item, 20 min per walk, $65/h). It always returns the assumptions and a
  caveat, so nobody mistakes an estimate for a measurement.

**Not yet proven (fill this in before pitching)**
- No customer has used it on a live job. The 14 labelled defects are a smoke test, not a benchmark.
- Add 3 to 5 interviews with superintendents or project managers. Record: how long their last punch
  write-up took, how many items, what they use today (Procore, Fieldwire, spreadsheets), what they would pay.
  Put real quotes and numbers in `docs/validation.md`. Do not paste invented ones.
- Run one real walk with an outside inspector and score it with `make eval-live`.

## 2. Technical execution and reliability (20%)

- Resumable, idempotent pipeline; every external call is cached, timed and logged as a `ModelCall`.
- Meta Muse Spark Open (Baseten) is the reasoning path, with a logged fallback to Muse Spark 1.3.
- Body-size limits, upload rate limit, daily and total spend caps, nightly demo reset.
- New: `GET /readyz` returns 503 unless the database, ffmpeg, data directory and both model keys are in
  place, so a deploy cannot look healthy while unable to process a walk. `GET /healthz` stays a liveness check.
- The API, contract and new insights suites (186 tests) pass offline. A handful of ffmpeg-version-sensitive tests (byte-exact replays,
  faststart detection) fail on some ffmpeg builds; they pass on the pinned Docker image.

**Gemini-only mode** (`GEMINI_API_KEY`) is untested against the live API: the request and reply handling is
covered by mocked tests only. Run one real walk before relying on it.

**Known limits:** single process, two jobs at a time, SQLite on one volume. Fine for a pilot. See section 5.

## 3. Trust, safety and human oversight (15%)

The design rule is that **the AI proposes and a person decides anything uncertain.**

| Control | Where |
|---|---|
| Every item is checked against its own photo and the inspector's words | `pipeline/verify.py` |
| Failed checks go to "Needs a look" and are **not sent** until a person taps "Approve and send" | `dispatch/policy.py`, `items.py` |
| No outline beats a wrong outline: outlines ship only after a score and a second check (6 correct, 0 wrong in the blind rating) | `pipeline/ground_v2.py` |
| Re-walk never signs off on a doubtful photo; a false "fixed" claim goes to a person | `pipeline/closures.py` |
| Every change is logged with actor (human, model, system) | `Item.history` |
| Every model call is visible to the user: model, latency, cost, cache or fallback | "How this walk was processed" |

**New in this pass**
- `GET /api/projects/{id}/trust`: counts what the AI did alone vs. what a person decided, how many outlines
  were withheld, how many false "fixed" claims were caught, and `flagged_sent_without_approval`, which must
  be 0. That number is replayed from each item's own history, so it checks the invariant rather than asserting it.
- `GET /api/projects/{id}/audit.csv`: every event on every item, dismissed ones included. Cells that start
  with `=`, `+`, `-` or `@` are escaped so an inspector's words can't become a spreadsheet formula.
- `POST /api/walks/{id}/purge-footage`: deletes raw clips and extracted audio, keeps the evidence photos and
  the playback video. Walks capture bystanders' faces and voices; this is data minimisation.
- `DELETE /api/projects/{id}`: erases every row and file of a project (right to erasure). Disabled in the
  public demo.

**Still open:** no authentication or per-user roles, and no automatic face blurring. Do not run this on a real
site with other people's footage until sign-in and a retention policy exist. Say so in the pitch; judges
reward knowing the boundary.

## 4. Innovation and defensibility (15%)

- **Say it, point if subtle.** Voice carries the item; a held finger pins the exact spot. SAM 3.1 follows the
  hand, so the inspector never has to stage anything.
- **The closing loop.** Most tools stop at the list. Punchwalk re-checks the same spot on the re-walk with
  before/after photos. That is the part that moves final payment.
- **Honest grounding.** Locate with a language model, segment with SAM, score, verify, or show nothing.

**What is and isn't a moat.** The models are available to everyone, so they are not the moat. What compounds is
(a) a labelled walk dataset and eval harness per trade and region, (b) construction vocabulary and
trade-routing rules, (c) integrations (Procore, Fieldwire) that hold the item history, and (d) trust:
a verified record that has been right so far. Open weights also allow a private deployment for owners
who will not send footage to a third-party cloud.

## 5. Business viability and scalability (20%)

**Measured unit cost** (`README.md`, cost ledger): about $0.54 for a 105 s, 12-item first walk; $0.03 to $0.05
per item with grounding v2. Estimate, not measured: $1.80 to $2.20 for a 10-minute, 20-defect walk.

**Pricing hypothesis (to test in interviews, not a fact):**

| Plan | Price | Includes | Cost to serve | Gross margin |
|---|---|---|---|---|
| Per walk | $15 | one walk, up to 25 items, sent to trades | about $2 | about 85% |
| Site | $199 per month | up to 10 walks, re-walks free, Slack and email dispatch | up to about $20 | about 90% |
| Firm | custom | SSO, audit export, Procore/Fieldwire sync, private deployment | varies | by contract |

Even at 10x the estimated model cost the per-walk plan stays above 50% gross margin.

**Who buys first:** general contractors and owner's representatives closing out multi-unit residential and
commercial fit-out, where punch lists are large and repeated. The same loop also fits rental turnovers,
move-in/move-out reports and facility walkthroughs; test one adjacent market only after the first is paying.

**Sustainability path (no revenue needed yet).** (1) 3 paid pilots at a site price, (2) turn the economics
endpoint into the pilot's own ROI report, (3) land a Procore or Fieldwire integration as the distribution
channel.

**Scaling plan, in order**
1. Sign-in and per-organisation tenancy (blocker for any paying customer).
2. Postgres in place of SQLite; media to object storage.
3. A real queue and stateless workers (Fly machines or similar) in place of the in-process pool.
4. Per-organisation spend caps and usage metering, built on the existing `ModelCall` ledger.
5. Procore/Fieldwire sync.

## 6. Pitch and communication (10%)

Lead with the live demo: `punchwalk.fly.dev` on a phone, open the "Needs a look" item #003.

**90-second script**
1. *Problem (15 s).* "The punch walk takes minutes. The paperwork takes hours, and final payment waits on it."
2. *Demo (40 s).* Item #011: tap Play this moment. Item #003: "marked fixed, but the defect is still visible",
   caught with before/after photos. Trades screen: two items go to Slack.
3. *Trust (15 s).* "Anything the AI isn't sure of waits for a person. Here's the count that proves it:
   `flagged_sent_without_approval: 0`."
4. *Business (15 s).* "About 54 cents of model cost for a 12-item walk. The paperwork it replaces is hours."
5. *Ask (5 s).* "We want three construction teams to run one real walk each."

Say the limits yourself before a judge finds them: small test set, one-process pilot architecture, no sign-in yet.
