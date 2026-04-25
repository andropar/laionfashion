# Development Notes

Use this repo as the source of truth for code. Treat Raven as the data/compute runner.

Before changing scope or architecture, read `docs/agent_handoff.md`. It captures the project intent, non-goals, safety boundaries, and MVP sequence.

## Local Loop

1. Edit locally with normal Python tooling.
2. Run unit tests with synthetic data:

```bash
pytest
```

3. Commit and push from local.

No local machine is expected to have `/ptmp/rothj` or LAION tars. Code should either use synthetic fixtures, exported debug bundles, or fail with a clear message.

## Raven Loop

```bash
ssh raven
cd /u/rothj/laionfashion
git pull
pip install -e ".[dev,app,server]"
```

For GPU scripts:

```bash
python /u/rothj/laion_natural/scripts/start_as_slurm_job.py --gpu \
  /u/rothj/laionfashion/scripts/<script>.py
```

Use the SLURM helper defaults unless a script explicitly needs different resources. In non-interactive shell commands, call the helper by filepath rather than relying on the `startslurm` alias.

## MVP Bundle Pipeline

The recommended sequence for building a case-study-ready bundle:

### Step 1: Build a CLIP-reranked subset

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

Pipeline stages:
1. Scans ~200k captions with outfit-mode scoring (score >= 2.5).
2. Collects ~1000 caption-matched candidates with thumbnails.
3. Scores all candidates with CLIP ViT-B-32 against person-in-outfit vs. product-only prompts.
4. Exports the top 200 by CLIP outfit score.
5. Writes diagnostics: `filter_summary.json`, `accepted_captions.csv`, `rejected_captions.csv`.

### Step 2: Build UMAP projection

```bash
python scripts/02_build_projection.py <bundle>
```

Writes `projection.parquet`. Auto-selects UMAP for >= 15 images.

### Step 3: Compute CLIP prompt-direction style axes

```bash
python scripts/05_build_clip_axes.py <bundle>
```

Encodes bundle thumbnails and 6 axis prompt pairs with CLIP, writes `axis_scores.parquet`:
- Colorful vs neutral
- Streetwear vs classic
- Sporty vs dressy
- Minimalist vs maximalist
- Polished vs rough
- Formal vs casual

These are exploratory prompt-direction axes, not ground-truth labels.

### Step 4: Generate a review contact sheet

```bash
python scripts/04_make_review_contact_sheet.py <bundle>
```

Writes a self-contained `contact_sheet.html` for visual quality review.

### Step 5: Browse with Streamlit

```bash
streamlit run app/streamlit_app.py
```

The app auto-loads projection, axes, and manifest. Features:
- 2D embedding map with axis coloring
- Nearest-neighbor retrieval
- Top/bottom examples per axis with prompt labels
- Dataset info panel (CLIP reranking stats, filtering diagnostics)

## Raven / Local Agent Protocol

Local Claude Code:
- Owns code edits, package structure, app implementation, and local tests.
- Should work without `/ptmp/rothj`.
- Pushes changes to GitHub when ready for server validation.

Raven agent:
- Pulls the latest code after the user says it is ready.
- Runs inventory scripts, debug subset builders, and SLURM jobs.
- Reports what worked, what failed, and what local code should change next.
- LAION-derived image outputs stay private/local.

Handoff format:

```text
Please implement <specific feature>. Context from Raven:
- Bundle path / contents:
- Observed issue:
- Expected behavior:
- Validation command:
```

## Data Outputs

Generated outputs belong under:

```text
scripts/outputs/<script_name>/<timestamp>_<id>/
```

These outputs are ignored by git. For local UI development, copy only small derived bundles from Raven. Do not expose raw LAION images publicly.

## Legacy Scripts

- `scripts/00_inventory.py` — LAION-natural shard inventory (Raven only).
- `scripts/03_build_demo_axes.py` — Proxy axes from PCA + caption keywords. Superseded by `05_build_clip_axes.py` for bundles with CLIP access.
