# Scale-up commands for Raven

Copy-paste ready commands for building larger bundles.

## Prerequisites

```bash
cd /u/rothj/laionfashion
git pull
pip install -e ".[dev,app,server]"
pytest
```

## Option A: 5k images (recommended first scale-up)

```bash
# Step 1: Build CLIP-reranked subset (5k from 25k candidates, scan 1M captions)
python /u/rothj/laion_natural/scripts/start_as_slurm_job.py \
  /u/rothj/laionfashion/scripts/01_build_debug_subset.py \
  --n-images 5000 \
  --n-candidates 25000 \
  --candidate-scan 1000000 \
  --thumbnail-size 160 \
  --detection-image-size 768 \
  --selection-mode outfit \
  --clip-rerank

# Step 2: Extract garments with DETR
python scripts/06_extract_garments.py <bundle>

# Step 3: Embed garment crops
python scripts/07_embed_garments.py <bundle>

# Step 4: Build UMAP projection
python scripts/02_build_projection.py <bundle>

# Step 5: Compute CLIP style axes
python scripts/05_build_clip_axes.py <bundle>

# Step 6: Build search index
python scripts/12_build_search_index.py <bundle>

# Step 7: Cluster and label
python scripts/14_cluster_and_label.py <bundle> --n-clusters 15

# Step 8: Run evaluation
python scripts/08_evaluate_retrieval.py <bundle>

# Step 9: Generate review sheets
python scripts/04_make_review_contact_sheet.py <bundle>
python scripts/09_garment_review.py <bundle>

# Step 10: Compare embeddings (CLIP vs FashionCLIP)
python scripts/13_compare_embeddings.py <bundle> --models clip-vit-b-32,fashionclip

# Step 11: Validate and pack for local dev
python scripts/11_validate_portable_bundle.py <bundle>
python scripts/10_pack_bundle.py <bundle>
```

## Option B: 20k images (strong demo)

Same as above but with:
```bash
--n-images 20000 \
--n-candidates 100000 \
--candidate-scan 5000000 \
```

This will take significantly longer for CLIP reranking (100k images to score).
Consider running steps 1-3 with `--gpu` via SLURM.

## Option C: 50k images (large-scale)

```bash
--n-images 50000 \
--n-candidates 250000 \
--candidate-scan 10000000 \
```

At this scale, consider:
- UMAP may need `n_neighbors=30` or higher
- FAISS index should still be fine with IndexFlatIP up to ~100k
- Garment detection will take hours — run as a SLURM job
- Contact sheets will be very large — use `--max-images 500`

## GPU jobs

For CLIP-intensive steps (reranking, garment embedding, axis computation):

```bash
python /u/rothj/laion_natural/scripts/start_as_slurm_job.py --gpu \
  /u/rothj/laionfashion/scripts/<script>.py <args>
```

## Expected timings (rough, on SLURM)

| Step | 200 images | 5k images | 20k images |
|------|-----------|-----------|------------|
| Caption scanning | ~1 min | ~10 min | ~40 min |
| CLIP reranking | ~2 min | ~15 min | ~60 min |
| Garment detection | ~9 min | ~3 hr | ~12 hr |
| Garment embedding | ~1 min | ~5 min | ~20 min |
| UMAP projection | <1 min | ~2 min | ~10 min |
| CLIP axes | ~1 min | ~5 min | ~20 min |

## After building

Transfer the packed bundle to local:

```bash
scp raven:/u/rothj/laionfashion/<bundle_name>.tar.gz .
tar xzf <bundle_name>.tar.gz
python scripts/11_validate_portable_bundle.py <bundle_dir>
streamlit run app/streamlit_app.py
```
