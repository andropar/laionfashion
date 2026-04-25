# Progress tracker

Status key: DONE / PARTIAL / TODO / BLOCKED

## Section 19.1 — Technical completion

| Item | Status | Notes |
|------|--------|-------|
| Curated fashion/person subset | PARTIAL | 200 images, need 5k-50k |
| Reproducible filtering scripts | DONE | 01_build_debug_subset.py with outfit mode + CLIP rerank |
| Thumbnail/contact-sheet inspection | DONE | 04_make_review_contact_sheet.py |
| CLIP or OpenCLIP embeddings | DONE | ViT-B-32/laion400m_e31 |
| Fashion-specific or alternative embedding | TODO | Need FashionCLIP comparison |
| Prompt-axis scoring | DONE | 6 axes via 05_build_clip_axes.py |
| Vector search index | TODO | Need FAISS or Annoy index |
| 2D embedding projection | DONE | UMAP via 02_build_projection.py |
| Cluster exemplars | TODO | Need cluster labels + exemplar selection |
| Interactive explorer | PARTIAL | Streamlit works but needs polish |
| Documented failure cases | TODO | |
| Clean config-driven pipeline | PARTIAL | Scripts work, no YAML config |
| Clear repo structure | DONE | |
| Minimal reproducibility instructions | DONE | README + docs |

## Section 19.2 — Data-product completion

| Item | Status | Notes |
|------|--------|-------|
| Select outfit/image | DONE | |
| View similar/style-neighbor images | DONE | Nearest neighbors panel |
| See where it lies in style space | DONE | Embedding map |
| At least 5 interpretable style axes | DONE | 6 CLIP axes |
| Compare >= 2 embedding approaches | TODO | Only CLIP, need FashionCLIP/DINOv2 |
| Show model limitations/failures | TODO | |
| Generate portfolio-worthy screenshots | TODO | UI needs polish |

## Section 19.3 — Communication completion

| Item | Status | Notes |
|------|--------|-------|
| Case study page on jroth.space | TODO | |
| Hero screenshot | TODO | |
| Visual pipeline diagram | TODO | |
| Style-space atlas figure | TODO | |
| 60-90 second demo video | TODO | |
| GitHub README with screenshots | PARTIAL | README exists, no screenshots |
| Concise technical explanation | PARTIAL | |
| Limitations / ethics section | TODO | |
| "What this demonstrates" section | TODO | |

## Section 19.4 — Evaluation completion

| Item | Status | Notes |
|------|--------|-------|
| Pairwise human labels | TODO | |
| Prompt-axis vs probe comparison | TODO | |
| Retrieval sanity checks | PARTIAL | eval harness exists but not run at scale |
| Model comparison table | TODO | |
| Documented failure taxonomy | TODO | |

## Section 19.5 — Portfolio completion

| Item | Status | Notes |
|------|--------|-------|
| Homepage case study | TODO | |
| GitHub repo | DONE | |
| Demo video | TODO | |
| Image-heavy artifact | TODO | |
| Technical writeup | TODO | |

## Section 19.6 — "Wow" completion

| Item | Status | Notes |
|------|--------|-------|
| At least one wow moment | TODO | Best candidate: style map with axis coloring |

## What I can do locally (no Raven)

- Build polished React/Next.js frontend or upgrade Streamlit
- Build case study page
- Create pipeline/architecture diagrams
- Write technical report
- Add FAISS vector search index
- Add FashionCLIP embedding support (if weights downloadable)
- Add cluster labels from UMAP
- Create failure case documentation
- Polish everything for screenshots
- Write dataset/model card
- Add limitations/ethics section
- Create visual atlas layout

## What needs Raven

- Scale to 5k-50k images
- Run embeddings on larger dataset
- Run garment detection at scale
- Run evaluation at scale
- Generate human pairwise labels (needs annotation UI)
