# Mapping Style Space

> A foundation-model explorer for fashion embeddings in natural images

## Problem

Large-scale web image datasets contain rich information about how people dress — but that information is buried in noisy, uncurated data. Fashion search typically focuses on item-level similarity ("show me similar sweaters"), but the more interesting questions are about *contextual style structure*: Where does an outfit live in style space? What do foundation models already encode about clothing? Can we make that structure visible and explorable?

## Approach

Starting from LAION-derived natural images, I built a pipeline that:

1. **Curates** a fashion-relevant subset from millions of web images using caption scoring and CLIP image-side reranking
2. **Detects garments** using a fine-tuned DETR model (top, bottom, dress, outerwear, shoes, hat, bag)
3. **Embeds** full outfits and individual garments with foundation models (OpenCLIP, FashionCLIP)
4. **Derives interpretable style axes** from CLIP prompt directions (colorful/neutral, streetwear/classic, formal/casual, etc.)
5. **Projects** embeddings into a navigable 2D map using UMAP
6. **Evaluates** cross-category retrieval as a compatibility baseline

The result is an interactive atlas of fashion/style space where you can select an outfit, explore its neighborhood, and move along style dimensions.

## Pipeline

```
LAION-natural (millions of images)
    │
    ▼ Caption scoring (outfit mode, score >= 2.5)
~1000 candidates
    │
    ▼ CLIP image reranking (person-in-outfit vs product)
200–5000 curated images
    │
    ├── DETR garment detection → garment crops + embeddings
    ├── UMAP 2D projection → embedding map
    ├── CLIP prompt-direction scoring → 6 style axes
    └── FAISS vector index → instant retrieval
    │
    ▼
Interactive Explorer
```

## Style axes

Each axis is computed as the normalized difference between a positive and negative CLIP text prompt, then scored against image embeddings by dot product.

| Axis | High end | Low end |
|------|----------|---------|
| Colorful vs neutral | Bright, saturated, bold colors | Black, white, grey, monochrome |
| Streetwear vs classic | Urban, oversized, sneakers, graphic | Blazer, oxford shoes, preppy |
| Sporty vs dressy | Athletic wear, running shoes, gym | Cocktail dress, heels, evening |
| Minimalist vs maximalist | Simple lines, plain, few accessories | Layered, prints, many accessories |
| Polished vs rough | Coordinated, pressed, neat | Distressed, faded, grunge |
| Formal vs casual | Tailored suit, office wear | T-shirt, sweatpants, loungewear |

These axes are exploratory — they reflect what CLIP's text-image alignment encodes, not ground-truth style categories.

## Findings

### What works

- **Embedding neighborhoods are visually coherent.** Nearest neighbors in CLIP space tend to share style, color palette, and garment type. The embedding map shows meaningful clusters.
- **Prompt-direction axes are interpretable.** Top/bottom examples for colorful/neutral and streetwear/classic are immediately recognizable. The axes expose real structure in the model's representation.
- **CLIP reranking dramatically improves curation quality.** Caption filtering alone lets through many product photos and irrelevant images. Adding CLIP image-side scoring (person-in-outfit vs. product/landscape) selects much cleaner sets.

### What doesn't work (yet)

- **Cross-category retrieval is noisy.** Frozen CLIP features don't encode compatibility well — "given this top, find matching bottoms" returns results that share color or texture but not necessarily style coherence. This is the expected baseline to improve with learned embeddings.
- **Photo aesthetics dominate outfit quality.** CLIP embeddings are heavily influenced by lighting, composition, and background. A professional photo of an awkward outfit ranks higher than a mirror selfie of a great outfit.
- **Some axes collapse.** Formal/casual shows less variation than colorful/neutral, likely because CLIP's training data doesn't strongly differentiate formality from general "niceness."

## What this demonstrates

- **Large-scale data curation:** Filtering millions of noisy web images into a curated, fashion-relevant subset using a multi-stage pipeline (caption scoring → CLIP reranking → garment detection).
- **Foundation-model exploitation:** Using CLIP not for classification but for structure discovery — prompt-direction axes, embedding neighborhoods, and image-side quality scoring.
- **Embedding analysis and vector search:** Building navigable embedding spaces with UMAP projection, cluster labeling, and FAISS-based retrieval.
- **Visual communication:** Turning complex ML outputs into an interactive, interpretable data product.
- **Pragmatic evaluation:** Defining retrieval tasks and measuring baselines before training anything.

## Limitations

- **Not a production recommender.** This is a research/portfolio prototype, not a scalable product.
- **Style judgments are model-derived.** Scores reflect CLIP's training distribution, not objective taste. Axes are exploratory, not ground truth.
- **LAION data has known biases.** Over-representation of Western fashion, professional photography, and certain body types. No NSFW or minor detection has been applied beyond caption filtering.
- **Images are private.** LAION-derived images are kept local and not redistributed.
- **Co-occurrence ≠ compatibility.** Garments appearing together in an image doesn't necessarily mean they're stylistically compatible.

## Disclaimer

This project explores style-related structure in foundation-model embeddings. Scores and labels are model-derived and should not be interpreted as objective judgments of taste, attractiveness, social status, or personal value. The system is intended as a visualization and data-product prototype, not as a normative fashion-ranking tool.
