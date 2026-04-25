# Mapping Style Space

> A foundation-model explorer for fashion embeddings in natural images

<!-- TODO: Add hero screenshot here -->
<!-- ![Explorer screenshot](assets/screenshots/hero.png) -->

## What this is

An interactive embedding explorer that uses foundation models (CLIP, FashionCLIP) to map, search, and visualize clothing styles in natural images. Starting from LAION-derived web photos, the pipeline filters for people wearing visible clothing, extracts garment-level features, derives interpretable style axes, and presents the result as a navigable style-space atlas.

**This is a research/portfolio prototype**, not a production fashion recommender. Scores and labels are model-derived and exploratory, not objective judgments of taste.

## What it demonstrates

- Large-scale data curation from noisy web images
- Foundation-model exploitation (CLIP/FashionCLIP for scoring, axes, retrieval)
- Embedding analysis with interpretable prompt-direction axes
- Vector search and cross-category garment retrieval
- Interactive data-product design
- Pragmatic ML evaluation (hold-out retrieval, failure taxonomy)

## Pipeline

```
LAION-natural (millions of images)
    │
    ▼ Caption scoring (outfit mode, score >= 2.5)
~1000+ candidates
    │
    ▼ CLIP image reranking (person-in-outfit vs product)
200–5000+ curated images
    │
    ├── DETR garment detection → garment crops + CLIP embeddings
    ├── UMAP 2D projection → embedding map
    ├── CLIP prompt-direction scoring → 6 style axes
    ├── FAISS vector index → instant retrieval
    └── KMeans clustering → style neighborhoods
    │
    ▼
Interactive Explorer
```

## Style axes

| Axis | High end | Low end |
|------|----------|---------|
| Colorful vs neutral | Bright, saturated, bold | Black, white, monochrome |
| Streetwear vs classic | Urban, oversized, sneakers | Blazer, oxford shoes, preppy |
| Sporty vs dressy | Athletic, running shoes, gym | Cocktail dress, heels |
| Minimalist vs maximalist | Simple, plain, few accessories | Layered, prints, busy |
| Polished vs rough | Coordinated, neat, pressed | Distressed, faded, grunge |
| Formal vs casual | Tailored suit, office wear | T-shirt, sweatpants |

## Quick start

### Local development (from a portable bundle)

```bash
git clone git@github.com:andropar/laionfashion.git
cd laionfashion
pip install -e ".[dev,app]"
pytest

# Unpack a bundle (get from Raven)
tar xzf <bundle_name>.tar.gz
python scripts/11_validate_portable_bundle.py <bundle_dir>

# Launch explorer
streamlit run app/streamlit_app.py
```

### Build a bundle on Raven

```bash
pip install -e ".[dev,app,server]"

# Full pipeline (see docs/scale_up_commands.md for larger runs)
python scripts/01_build_debug_subset.py \
  --n-images 200 --n-candidates 1000 --candidate-scan 200000 \
  --thumbnail-size 160 --detection-image-size 768 \
  --selection-mode outfit --clip-rerank

python scripts/06_extract_garments.py <bundle>
python scripts/07_embed_garments.py <bundle>
python scripts/02_build_projection.py <bundle>
python scripts/05_build_clip_axes.py <bundle>
python scripts/12_build_search_index.py <bundle>
python scripts/14_cluster_and_label.py <bundle>
python scripts/08_evaluate_retrieval.py <bundle>
python scripts/10_pack_bundle.py <bundle>
```

## Documentation

- [Case study](docs/case_study.md) — Project narrative and findings
- [Development notes](docs/development.md) — Pipeline workflow
- [Scale-up commands](docs/scale_up_commands.md) — Raven commands for larger bundles
- [Agent handoff](docs/agent_handoff.md) — Project brief and direction
- [Dataset card](docs/data_card.md) — Data sourcing, biases, safety
- [Model card](docs/model_card.md) — Foundation models used

## Limitations

- **Not a recommender.** Research/portfolio prototype only.
- **Style judgments are model-derived.** Axes reflect CLIP's training distribution, not objective taste.
- **LAION data has known biases.** Western-centric, over-representation of professional photography.
- **Co-occurrence ≠ compatibility.** Garments in the same image aren't necessarily stylistically compatible.
- **Images are private.** LAION-derived images are not redistributed.

## Disclaimer

This project explores style-related structure in foundation-model embeddings. Scores and labels are model-derived and should not be interpreted as objective judgments of taste, attractiveness, social status, or personal value.
