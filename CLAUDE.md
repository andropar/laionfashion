# CLAUDE.md — Fashion Embedding Explorer

## CRITICAL: Completion requirements

**Read `completion_plan.md` section 19 before doing anything.** All completion criteria must be fulfilled. Track progress in `progress.md`. Do not stop until done.

## Project overview

A garment-level outfit representation workbench built on LAION-natural image data. The goal is a polished, portfolio-ready fashion embedding explorer that demonstrates large-scale data curation, foundation-model exploitation, embedding analysis, vector search, and visual communication.

Read `docs/agent_handoff.md` before making design or scope decisions.

## Key documentation

- `completion_plan.md` — Full project spec with completion criteria (section 19)
- `progress.md` — Progress tracking
- `docs/agent_handoff.md` — Project brief, priorities, safety boundaries
- `docs/development.md` — Pipeline workflow, Raven commands

### How to update docs

- **agent_handoff.md**: source of truth for *what* and *why*. Update when direction changes.
- **development.md**: source of truth for *how*. Update when pipeline changes.
- **README.md**: public-facing summary. Keep concise.
- **progress.md**: track what's done vs what remains against completion_plan.md.

## Development commands

```bash
pip install -e ".[dev,app]"     # local dev
pip install -e ".[dev,app,server]"  # Raven (includes umap-learn, torch, etc.)
pytest                           # run all tests
streamlit run app/streamlit_app.py  # launch explorer
```

## Code structure

- `src/laionfashion/` — Library code
- `scripts/` — CLI pipeline scripts (01–11)
- `app/` — Streamlit explorer
- `tests/` — Pytest suite (all local-friendly with mocks)

## Conventions

- Tests must stay lightweight. Use mock scorers for CLIP-dependent logic.
- Scripts: `scripts/0N_verb_noun.py <bundle_dir> [options]`
- Bundle outputs: `scripts/outputs/` (gitignored)
- LAION images: private/local only
- Plot labels: capitalize only first word
- Style axes: exploratory, not ground truth
- Local code must work without `/ptmp/rothj`

## Raven execution

```bash
python /u/rothj/laion_natural/scripts/start_as_slurm_job.py \
  /u/rothj/laionfashion/scripts/<script>.py [args]
```

## Current state (2026-04-25)

Phase 1 complete: caption filtering, CLIP reranking, UMAP projection, 6 CLIP prompt-direction axes, Streamlit explorer.

Phase 2 in progress: garment detection (DETR), garment CLIP embeddings, cross-category retrieval baseline, evaluation harness, portable bundles.

Major gaps vs completion_plan.md section 19:
- Scale: need 5k-50k images, currently 200
- No FashionCLIP or alternative embedding comparison
- No vector search index (FAISS/Annoy)
- No human labels / pairwise evaluation
- No case study page
- No demo video
- No visual atlas figure
- No dataset/model card
- Explorer UI needs polish for portfolio screenshots
