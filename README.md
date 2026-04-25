# Fashion Embedding Explorer

Local-friendly scaffold for a private LAION fashion/outfit embedding explorer.

The goal is to develop code locally, push to GitHub, then pull on Raven and run data jobs against the LAION-natural subset. The implementation target is outfit-level exploration: curated image subset, CLIP-reranked selection, prompt-direction style axes, 2D projection, and a local/private explorer app.

## Repository Workflow

Future coding agents should start by reading `docs/agent_handoff.md` for the project intent, safety boundaries, MVP milestones, and Raven/local collaboration protocol.

Local development:

```bash
git clone git@github.com:andropar/laionfashion.git
cd laionfashion
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,app]"
pytest
```

## MVP Bundle Workflow

The full pipeline to produce an MVP-quality bundle:

### 1. Build a CLIP-reranked debug subset (Raven)

Collect caption-matched candidates, score with CLIP, export the top N:

```bash
python /u/rothj/laion_natural/scripts/start_as_slurm_job.py \
  /u/rothj/laionfashion/scripts/01_build_debug_subset.py \
  --n-images 200 \
  --n-candidates 1000 \
  --candidate-scan 200000 \
  --thumbnail-size 160 \
  --selection-mode outfit \
  --clip-rerank
```

This scans ~200k captions, collects ~1000 outfit-mode matches, scores all with CLIP ViT-B-32, and exports the top 200 by person-in-outfit score.

### 2. Build UMAP projection

```bash
python scripts/02_build_projection.py <bundle>
```

Auto-selects UMAP for bundles >= 15 images. Writes `projection.parquet`.

### 3. Compute CLIP prompt-direction style axes

```bash
python scripts/05_build_clip_axes.py <bundle>
```

Encodes thumbnails and axis prompts with CLIP ViT-B-32, scores each image against 6 prompt-direction axes (colorful/neutral, streetwear/classic, sporty/dressy, minimalist/maximalist, polished/rough, formal/casual). Writes `axis_scores.parquet`.

### 4. Generate a contact sheet for review

```bash
python scripts/04_make_review_contact_sheet.py <bundle>
```

Produces a self-contained HTML contact sheet with embedded thumbnails.

### 5. Browse with the Streamlit app

```bash
pip install -e ".[app]"
streamlit run app/streamlit_app.py
```

Point the sidebar at the bundle directory. The app loads projection, axes, and manifest automatically.

To copy a bundle from Raven for local browsing:

```bash
scp -r raven:/u/rothj/laionfashion/scripts/outputs/01_build_debug_subset/<bundle> \
  ./scripts/outputs/01_build_debug_subset/
```

## Data Assumptions

On Raven, the default data roots are:

- LAION-natural subset: `/ptmp/rothj/cstims_laion_natural_subset`
- LAION-natural feature memmaps: `/ptmp/rothj/cstims_laion_natural_subset_memmaps`

The subset consists of tar shards with paired `.jpg` and `.json` entries. Keep generated image outputs local/private unless licensing and hosting are explicitly handled.

## Raven / Local Agent Protocol

1. Local Claude Code edits, tests with synthetic/local fixtures, commits, and pushes.
2. The Raven agent pulls the latest code, runs data jobs, inspects outputs, and summarizes findings.
3. The Raven agent reports concrete next code requests for local Claude Code.
4. Repeat until the MVP has a useful curated subset, embeddings, projection, style axes, and explorer.

Raven is the source of truth for data availability and job results. Local development should avoid assuming `/ptmp/rothj` exists and should work against exported debug bundles or synthetic tests.

## Current Scope

Outfit-level embedding exploration with CLIP-based curation and style axes. No segmentation, no model training, no public image hosting. The immediate focus is producing a visually compelling, screenshot-ready explorer for a case-study write-up.
