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
/u/rothj/conda-envs/laion/bin/python scripts/00_inventory.py
python /u/rothj/laion_natural/scripts/start_as_slurm_job.py \
  /u/rothj/laionfashion/scripts/01_build_debug_subset.py --n-images 1000
```

For GPU scripts later:

```bash
python /u/rothj/laion_natural/scripts/start_as_slurm_job.py --gpu \
  /u/rothj/laionfashion/scripts/<script>.py
```

Use the SLURM helper defaults unless a script explicitly needs different resources. In non-interactive shell commands, call the helper by filepath rather than relying on the `startslurm` alias.

## Raven / Local Agent Protocol

The intended workflow is split by responsibility.

Local Claude Code:

- Owns code edits, package structure, app implementation, and local tests.
- Should work without `/ptmp/rothj` by using synthetic fixtures or exported debug bundles.
- Pushes changes to GitHub when ready for server validation.

Raven agent:

- Pulls the latest code after the user says it is ready.
- Runs inventory scripts, debug subset builders, and SLURM jobs on the cluster.
- Reads generated manifests/logs/results and reports what worked, what failed, and what local code should change next.
- Does not assume public image hosting is allowed; LAION-derived image outputs stay private/local.

Suggested handoff format from Raven to local:

```text
Please implement <specific feature>. Context from Raven:
- Data/output path:
- Observed issue/result:
- Expected behavior:
- Validation command:
```

## Data Outputs

Generated outputs belong under:

```text
scripts/outputs/<script_name>/<timestamp>_<id>/
```

These outputs are ignored by git. For local UI development, copy only small derived bundles: metadata tables, thumbnails, embeddings/projections, and indices. Do not expose raw LAION images publicly.

## Running the Streamlit App

```bash
pip install -e ".[app]"
streamlit run app/streamlit_app.py
```

The app works entirely against exported debug bundles — no Raven paths required. Enter the bundle directory path in the sidebar (or point at `scripts/outputs/01_build_debug_subset/` to pick from sub-bundles).

To get a bundle from Raven, copy the output directory (e.g. via `scp -r raven:/u/rothj/laionfashion/scripts/outputs/01_build_debug_subset/<timestamp>_<id> ./scripts/outputs/01_build_debug_subset/`) to your local checkout.

## First MVP Bias

Start with full-image/outfit-level retrieval. Add person crops, garment parsing, and compatibility only after the basic map and nearest-neighbor demo are compelling.

