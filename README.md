# Fashion Embedding Explorer

Local-friendly scaffold for a private LAION fashion/outfit embedding explorer.

The goal is to develop code locally, push to GitHub, then pull on Raven and run data jobs against the LAION-natural subset. The first implementation target is outfit-level exploration: curated image subset, existing foundation-model embeddings, prompt/style axes, 2D projection, and a small local/private explorer.

## Repository Workflow

Local development:

```bash
git clone git@github.com:andropar/laionfashion.git
cd laionfashion
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,app]"
pytest
```

Raven execution:

```bash
cd /u/rothj/laionfashion
git pull
/u/rothj/conda-envs/laion/bin/python scripts/00_inventory.py
python /u/rothj/laion_natural/scripts/start_as_slurm_job.py \
  /u/rothj/laionfashion/scripts/01_build_debug_subset.py --n-images 1000
```

Use `/u/rothj/laion_natural/scripts/start_as_slurm_job.py` directly in scripted commands. It uses the `laion` conda environment by default.

## Raven / Local Agent Protocol

This project is meant to be a back-and-forth between local code development and Raven-side data execution.

1. Local Claude Code edits, tests with synthetic/local fixtures, commits, and pushes.
2. The Raven agent pulls the latest code, runs inventory/debug/SLURM jobs, inspects outputs, and summarizes findings.
3. The Raven agent reports concrete next code requests for local Claude Code.
4. Repeat until the MVP has a useful debug subset, embeddings/projection, and explorer.

Raven is the source of truth for data availability and job results. Local development should avoid assuming `/ptmp/rothj` exists and should work against exported debug bundles or synthetic tests.

## Data Assumptions

On Raven, the default data roots are:

- LAION-natural subset: `/ptmp/rothj/cstims_laion_natural_subset`
- LAION-natural feature memmaps: `/ptmp/rothj/cstims_laion_natural_subset_memmaps`
- Memmap metadata: `/ptmp/rothj/cstims_laion_natural_subset_memmaps/_metadata.pkl`

The subset consists of tar shards with paired `.jpg` and `.json` entries. Metadata currently includes captions, URLs, dimensions, hashes, and CLIP similarity, but sampled records did not expose NSFW or aesthetic scores. Keep generated image outputs local/private unless licensing and hosting are explicitly handled.

## First Debug Pipeline

Inventory:

```bash
/u/rothj/conda-envs/laion/bin/python scripts/00_inventory.py
```

Build a small fashion-caption-biased debug subset:

```bash
python /u/rothj/laion_natural/scripts/start_as_slurm_job.py \
  /u/rothj/laionfashion/scripts/01_build_debug_subset.py \
  --n-images 1000 \
  --candidate-scan 50000 \
  --feature-key openclip_vit_l_14_quickgelu_metaclip_fullcc.ln_post
```

Outputs are written to `scripts/outputs/<script_name>/<timestamp>_<id>/` and ignored by git.

## Current Scope

This scaffold intentionally does not start with segmentation or model training. The immediate next step is to produce a visually inspectable debug subset and confirm that nearest neighbors/style axes are interesting enough before adding garment-level logic.

