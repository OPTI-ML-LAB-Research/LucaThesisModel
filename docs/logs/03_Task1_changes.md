# T01 — Changes vs T05 (Groundwork closing delta)

This delta is **doc-only**. No code changed. Drop on top of your existing
project (overwrite when prompted).

## Files

| Path | Status | Purpose |
|---|---|---|
| `docs/sota_review.md` | NEW (replaces stub) | 213-line audit of HNKHSV.pptx + the 10 prior baseline models. Articulates the precise gap this MVP fills. |
| `docs/GROUNDWORK_GUIDE.md` | NEW | Step-by-step Windows reproduction recipe for everything T02–T05 produced. |
| `docs/GROUNDWORK_SUMMARY.md` | NEW | Concise handover doc — paste this at the start of every subsequent chat. |

## How to apply on Windows

```powershell
cd E:\Project\KhoaLuanCourse
Copy-Item -Recurse -Force T01_changes\* Raman-Physics-AI-v2\
```

## What this delta closes

The Groundwork phase (T01–T05). After applying this delta you have:

* Working scaffold (T02) — 90 files, 34 directories
* Verified dataloader + 3 split schemes (T03)
* Verified preprocessing pipeline + cache (T04)
* Verified BondMapper + seed DB (T05)
* SOTA review (T01)
* Reproduction guide and handover summary (T01 doc-only additions)

Read order for the next chat:
1. `docs/GROUNDWORK_SUMMARY.md` (this is the most important — paste at start of next chat)
2. `docs/sota_review.md` (background)
3. `docs/GROUNDWORK_GUIDE.md` (only if running the build)
