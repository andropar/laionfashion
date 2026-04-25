# CLAUDE.md — Fashion Embedding Explorer

## Project overview

A garment-level outfit representation workbench built on LAION-natural image data. Phase 1 (full-image explorer with CLIP axes) is complete. Phase 2 is garment-level parsing, cross-category retrieval, and learned embeddings.

Read `docs/agent_handoff.md` before making design or scope decisions. It has the project intent, non-goals, safety boundaries, and current phase priorities.

## Key documentation

- `docs/agent_handoff.md` — Durable project brief. Read first. Update when direction changes.
- `docs/development.md` — Local/Raven workflow, MVP pipeline steps, legacy scripts.
- `README.md` — Setup, run commands, pipeline overview.

### How to use and update docs

- **agent_handoff.md** is the source of truth for *what* and *why*. Update it when project goals, priorities, or constraints change. Do not let it become stale — if you change the strategic direction, update the handoff doc in the same commit.
- **development.md** is the source of truth for *how*. Update it when you add new scripts, change the pipeline, or modify the Raven workflow.
- **README.md** is the public-facing summary. Keep it concise. Update when the pipeline or setup instructions change.
- Do not duplicate information across docs. Cross-reference instead.

## Development commands

```bash
pip install -e ".[dev,app]"     # local dev
pip install -e ".[dev,app,server]"  # Raven (includes umap-learn, torch, etc.)
pytest                           # run all tests
streamlit run app/streamlit_app.py  # launch explorer
```

## Code structure

- `src/laionfashion/` — Library code (filtering, bundle loading, axes, projection, image scoring, review)
- `scripts/` — CLI scripts for data pipeline steps (01–05)
- `app/` — Streamlit explorer
- `tests/` — Pytest suite (all tests run without CLIP/GPU using mock scorers)

## Conventions

- Tests must stay lightweight and local-friendly. Use mock scorers (ConstantScorer, ListScorer) for CLIP-dependent logic.
- Scripts follow the pattern: `scripts/0N_verb_noun.py <bundle_dir> [options]`.
- Bundle outputs go under `scripts/outputs/` (gitignored).
- LAION-derived images are private/local. Do not expose publicly.
- Plot labels: capitalize only the first word.
- Style axes are exploratory prompt directions, not ground-truth labels. Use careful language.
- The Raven/local protocol is in `docs/agent_handoff.md`. Local code must work without `/ptmp/rothj`.

## Raven execution

Use the SLURM helper by filepath, not alias:

```bash
python /u/rothj/laion_natural/scripts/start_as_slurm_job.py \
  /u/rothj/laionfashion/scripts/<script>.py [args]
```

## Current state (2026-04-25)

Phase 1 complete: caption filtering, CLIP reranking, UMAP projection, 6 CLIP prompt-direction axes, Streamlit explorer with dataset info panel.

Phase 2 starting: garment-aware bundle format, person/garment detection, cross-category retrieval baseline, evaluation harness. See `docs/agent_handoff.md` Phase 2 for priorities.
