# Agent Handoff

This document is the durable project brief for future coding agents. Read it before making design or implementation decisions.

## One-Sentence Goal

Build a polished local/private MVP that uses existing foundation-model representations to explore fashion, outfits, and style structure in large-scale natural images. The project should show that messy visual data can be curated into an attractive, interpretable embedding-space explorer.

## Default Name

Use **Fashion Embedding Explorer** as the internal project name.

## Motivation

The project uses LAION-2B / LAION-natural style image data and existing server infrastructure to mine natural images for people wearing clothes. The interesting question is not only which garments look visually similar, but what pretrained visual and vision-language models already encode about clothing in context: outfit coherence, style, occasion, aesthetics, and compatibility proxies.

This is a portfolio and research-engineering project. Do not frame it as a production fashion recommender or as a model that learns objective taste. The intended case-study message is:

> I used large-scale web data, foundation models, embedding search, and interactive visualization to turn a messy image pool into a communicative data product.

## Core Positioning

The project should demonstrate data curation, representation work, interpretable axes, visualization, and product sense. Prefer foundation-model exploitation and clear communication over heavy custom training.

Use careful language:

- Good: "style-space projection", "foundation-model-derived compatibility signal", "context-aware clothing embedding", "weak aesthetic/style axis", "outfit coherence proxy", "retrieval based on visual and contextual similarity".
- Avoid: "objectively good/bad outfit", "fashion taste predictor", "beauty score", "attractiveness score", "body-type recommendation".

## Non-Goals And Safety Boundaries

Do not start by training a large model. Do not build a full recommender. Do not spend weeks on garment segmentation before validating the explorer. Do not rank people, bodies, attractiveness, gender expression, ethnicity, or personal characteristics. Do not expose raw LAION images publicly unless licensing, hosting, and terms-of-service questions are explicitly handled.

The first app should stay local/private. Treat real LAION-derived thumbnails as private artifacts unless told otherwise.

## Phase 1 MVP (done)

The full-image outfit explorer is working:

- Curated CLIP-reranked subset of people wearing visible clothing.
- Full-image embeddings from LAION-natural memmaps.
- UMAP 2D projection.
- 6 CLIP prompt-direction style axes (colorful/neutral, streetwear/classic, etc.).
- Streamlit app: embedding map, axis coloring, nearest neighbors, top/bottom examples.
- Contact sheet review, filter diagnostics, dataset info panel.

This is sufficient scaffolding. Do not spend more time polishing it.

## Phase 2: Garment-Level Representation Workbench

The real goal is to **learn/discover garment and outfit embeddings that organize fashion-relevant structure** — style, category, color, silhouette, formality, season — then ask whether compatibility emerges as a measurable behavior of that space.

Compatibility is an **evaluation target**, not a premature training signal.

### Priority order

1. **Garment-aware bundle format.** Add `garments.parquet`: outfit_id, garment_id, category, bbox/mask, crop_path, source image, CLIP embedding row. Start with bounding boxes and crops, not perfect segmentation.

2. **Compatibility baseline.** Use garment crops with frozen CLIP embeddings. Build cross-category retrieval: top → bottoms, dress → shoes, jacket → pants. It will likely be mediocre — that is the baseline to beat.

3. **Evaluation harness.** Hold out one garment from an outfit and rank candidate replacements from the same category. Metrics: recall@K, MRR, plus visual review sheets. Add hard negatives so the task is not trivial.

4. **Learned outfit/garment embeddings.** Start simple: frozen image encoder + small category-aware projection heads trained with contrastive/ranking loss on co-occurring garments. Only then consider bigger models or outfit-context transformers.

5. **Outfit builder demo.** Streamlit becomes: choose a garment or partial outfit, choose target category, compare CLIP vs. learned model recommendations, inspect why.

### Key caution

LAION natural images provide scale and diversity, but co-occurrence is a noisy weak label — "appeared together in an image" ≠ "stylistically compatible." Consider mixing in cleaner outfit datasets (Polyvore, DeepFashion, Fashionpedia) for supervision and evaluation.

### Representation-first philosophy

Do not train first. First define the retrieval/evaluation task. Otherwise you risk learning background, gender presentation, category shortcuts, or "street style photo" aesthetics instead of actual clothing structure.

The objective should stay exploratory and self-supervised at first: co-occurrence, same outfit, same source page/caption, visual/text consistency. Build tools to inspect axes, clusters, and retrieval behavior. Later add compatibility labels to validate or fine-tune.

## Current Server Reality

Raven has useful LAION assets:

- LAION-natural subset: `/ptmp/rothj/cstims_laion_natural_subset`
- LAION-natural feature memmaps: `/ptmp/rothj/cstims_laion_natural_subset_memmaps`
- Memmap metadata: `/ptmp/rothj/cstims_laion_natural_subset_memmaps/_metadata.pkl`

The natural subset contains tar shards with paired `.jpg` and `.json` records. Sampled metadata includes `caption`, `url`, image dimensions, hash/key fields, and CLIP similarity. Sampled metadata did not expose explicit NSFW, aesthetic, or language fields.

Precomputed embeddings are available as memmaps. Strong initial choices include OpenCLIP/MetaCLIP and DINOv2 feature spaces. Use one reliable embedding first; do not block the MVP on model comparison.

This repo is designed so local development does not require `/ptmp/rothj`. Use synthetic fixtures or exported debug bundles locally.

## Data Layer Shape

Use parquet for records and `.npy`, `.memmap`, or an ANN index for embeddings and retrieval. Keep schemas simple and debuggable.

Target record shape over time:

```json
{
  "image_id": "string",
  "image_path": "local/private path",
  "caption": "string",
  "width": 1024,
  "height": 768,
  "source_subset": "laion-natural",
  "person_score": 0.93,
  "outfit_visibility_score": 0.81,
  "embedding": "path or row id",
  "style_tags": ["minimal", "casual", "neutral"],
  "axis_scores": {
    "minimal_clean": 0.74,
    "formal_casual": -0.22,
    "coherent_awkward": 0.51,
    "colorful_neutral": -0.68
  }
}
```

Garment-level fields can be added later after the outfit-level explorer works.

## Prompt Axes

Implement prompt-direction scoring for a contrastive text-image model when the matching text encoder is available:

1. Encode positive prompt.
2. Encode negative prompt.
3. Normalize `positive - negative`.
4. Score image embeddings by dot product with the direction.

Candidate axes:

- "an elegant well-coordinated outfit" vs "an awkward badly matched outfit"
- "a minimalist clean outfit" vs "a busy cluttered outfit"
- "a formal business outfit" vs "a casual everyday outfit"
- "a streetwear outfit" vs "a classic formal outfit"
- "a colorful outfit" vs "a neutral monochrome outfit"
- "a modern stylish outfit" vs "an outdated unfashionable outfit"
- "a polished expensive-looking outfit" vs "a cheap poorly styled outfit"
- "a sporty outfit" vs "an elegant evening outfit"
- "a Scandinavian minimalist outfit" vs "a maximalist expressive outfit"

Treat these as exploratory axes, not ground truth.

## Filtering Roadmap

Stage 1 should use caption filtering, CLIP prompt retrieval, and existing LAION-natural scores to find likely fashion/outfit images.

Positive prompts include: "a full body photo of a person wearing an outfit", "a street style fashion photo", "a person showing their outfit", "a casual outfit photo", "a fashion photo of a person standing", "a person wearing everyday clothes", and "a full body portrait with visible clothes".

Negative concepts include close-up face portraits, groups of many people, children, babies, underwear, swimwear, nudity, violent images, medical images, cartoons, drawings, text documents, and product-only clothing on white backgrounds.

For the MVP, prefer one visible adult person, torso/full body visible, clear clothing, decent resolution, low NSFW risk, not crowded, and not heavy watermark/text.

Stage 2 adds person/outfit visibility filtering. Stage 3 adds garment detection/parsing only if needed.

## Compatibility Framing

Similarity is same-category or same-context retrieval: sweater to sweater, outfit to outfit. This can use embedding nearest neighbors.

Compatibility is cross-category or item-to-outfit retrieval: sweater to trousers, jacket to shoes, partial outfit to missing item suggestions. For the MVP, approximate this with co-occurrence in natural outfits, style-axis similarity, color harmony features, weak VLM labels, or tiny human-calibrated probes. Do not claim causal or objective compatibility.

## Explorer UI Direction

A strong demo eventually has query, neighbor retrieval, compatible items, a style map, a simple outfit builder, and a dataset/model debug panel. The immediate Streamlit version can be much smaller: bundle selector, thumbnail grid, selected query image, nearest neighbors, and basic caveats.

Plot labels should follow the user's plot rule: only the first word capitalized.

## Evaluation And Sanity Checks

Inspect top/bottom images for each axis, random nearest neighbors, style clusters, and failure cases. For retrieval, ask whether neighbors match style, color, category, occasion, and whether they are visually diverse enough.

For compatibility experiments, check for same-image leakage, background/pose overfitting, gender-presentation collapse, body judgment, and whether results are ranking people rather than clothes.

Bias and safety checks should look for children/minors, NSFW, swimwear/underwear, face-centric ranking, culture/class bias in "expensive" or "good" axes, offensive labels, and private-looking images.

## Case-Study Deliverables

The MVP should eventually produce screenshots of the explorer overview, query plus nearest neighbors, style embedding map, axis comparison, dataset pipeline, and failure/caveats panel.

Narrative:

> Large-scale natural image datasets contain rich information about clothing in context, but the structure is buried in noisy web data. I filtered a large natural image corpus for visible outfits, extracted foundation-model embeddings, derived interpretable style axes, and built an explorer for navigating similarity, compatibility proxies, and style clusters.

## Raven / Local Agent Protocol

Local Claude Code owns code edits, app implementation, and local tests. It should work without `/ptmp/rothj`.

Raven agent owns server inventory, data jobs, SLURM execution, generated-output inspection, and concrete next requests for local implementation.

Use this handoff format:

```text
Please implement <specific feature>. Context from Raven:
- Bundle path / contents:
- Observed issue:
- Expected behavior:
- Validation command:
```

Run SLURM helper commands by filepath, not by alias:

```bash
python /u/rothj/laion_natural/scripts/start_as_slurm_job.py /u/rothj/laionfashion/scripts/<script>.py
```

Use helper defaults unless a script explicitly needs different resources.

## Open Questions

Before overbuilding, clarify whether the primary unit is full image, person crop, or garment crop; whether real LAION images can ever be shown publicly; whether the target is portfolio, research prototype, product, or fun visualization; whether we want VLM labels or human labels; how careful we need to be around faces and identifiability; and what smallest result makes the project worth turning into a case study.

