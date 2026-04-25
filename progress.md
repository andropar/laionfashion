# Progress tracker

Status key: DONE / PARTIAL / TODO / BLOCKED(Raven)

## Section 19.1 — Technical completion

| Item | Status | Notes |
|------|--------|-------|
| Curated fashion/person subset | BLOCKED(Raven) | 200 images done, scripts ready for 5k-50k (see docs/scale_up_commands.md) |
| Reproducible filtering scripts | DONE | 01_build_debug_subset.py with outfit mode + CLIP rerank |
| Thumbnail/contact-sheet inspection | DONE | 04_make_review_contact_sheet.py |
| CLIP or OpenCLIP embeddings | DONE | ViT-B-32/laion400m_e31 |
| Fashion-specific or alternative embedding | DONE | models.py + 13_compare_embeddings.py (CLIP + FashionCLIP) |
| Prompt-axis scoring | DONE | 6 axes via 05_build_clip_axes.py |
| Vector search index | DONE | search.py + 12_build_search_index.py (FAISS IndexFlatIP) |
| 2D embedding projection | DONE | UMAP via 02_build_projection.py |
| Cluster exemplars | DONE | clusters.py + 14_cluster_and_label.py |
| Interactive explorer | DONE | Streamlit with tabs, garment view, cluster overlay |
| Documented failure cases | DONE | failures.py + 18_failure_report.py |
| Clean config-driven pipeline | DONE | configs/pipeline.yaml, run_full_pipeline.py |
| Clear repo structure | DONE | |
| Minimal reproducibility instructions | DONE | README + docs |

## Section 19.2 — Data-product completion

| Item | Status | Notes |
|------|--------|-------|
| Select outfit/image | DONE | |
| View similar/style-neighbor images | DONE | Nearest neighbors panel |
| See where it lies in style space | DONE | Embedding map with cluster overlay |
| At least 5 interpretable style axes | DONE | 6 CLIP axes |
| Compare >= 2 embedding approaches | DONE | models.py supports CLIP + FashionCLIP comparison |
| Show model limitations/failures | DONE | failures.py, failure_report.md |
| Generate portfolio-worthy screenshots | BLOCKED(Raven) | Need larger bundle for compelling map |

## Section 19.3 — Communication completion

| Item | Status | Notes |
|------|--------|-------|
| Case study page on jroth.space | PARTIAL | docs/case_study.md written, needs deployment |
| Hero screenshot | BLOCKED(Raven) | Need larger bundle |
| Visual pipeline diagram | DONE | assets/diagrams/pipeline.svg |
| Style-space atlas figure | BLOCKED(Raven) | Need larger bundle for meaningful clusters |
| 60-90 second demo video | TODO | Need screenshots first |
| GitHub README with screenshots | PARTIAL | README done, screenshots pending |
| Concise technical explanation | DONE | case_study.md |
| Limitations / ethics section | DONE | data_card.md, model_card.md, case_study.md |
| "What this demonstrates" section | DONE | README + case_study.md |

## Section 19.4 — Evaluation completion

| Item | Status | Notes |
|------|--------|-------|
| Pairwise human labels | PARTIAL | annotation.py + 15_generate + 16_evaluate + 17_viewer ready, need human to annotate |
| Prompt-axis vs probe comparison | BLOCKED(Raven) | Need annotations first |
| Retrieval sanity checks | DONE | evaluation.py + 08_evaluate_retrieval.py |
| Model comparison table | DONE | 13_compare_embeddings.py ready to run |
| Documented failure taxonomy | DONE | failures.py + 18_failure_report.py |

## Section 19.5 — Portfolio completion

| Item | Status | Notes |
|------|--------|-------|
| Homepage case study | PARTIAL | Content written, needs jroth.space deployment |
| GitHub repo | DONE | |
| Demo video | TODO | |
| Image-heavy artifact | BLOCKED(Raven) | Need larger bundle |
| Technical writeup | DONE | case_study.md |

## Section 19.6 — "Wow" completion

| Item | Status | Notes |
|------|--------|-------|
| At least one wow moment | BLOCKED(Raven) | Style map with 5k+ images + axis coloring + garment retrieval |

## What needs Raven to unblock

All remaining items are blocked on building a larger bundle (5k+ images). Run:

```bash
cd /u/rothj/laionfashion && git pull && pip install -e ".[dev,app,server]"

# Build 5k-image bundle
python /u/rothj/laion_natural/scripts/start_as_slurm_job.py \
  /u/rothj/laionfashion/scripts/01_build_debug_subset.py \
  --n-images 5000 --n-candidates 25000 --candidate-scan 1000000 \
  --thumbnail-size 160 --detection-image-size 768 \
  --selection-mode outfit --clip-rerank

# Then run full pipeline
python scripts/run_full_pipeline.py <bundle>

# Pack for local
python scripts/10_pack_bundle.py <bundle>
```

Then transfer to local, annotate pairwise labels via annotation_viewer.html,
take screenshots, record demo video.
