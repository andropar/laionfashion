# Dataset card

## Overview

| Field | Value |
|-------|-------|
| Name | Fashion Embedding Explorer subset |
| Source | LAION-natural (derived from LAION-2B) |
| Purpose | Research / portfolio prototype |
| License | Research use only; no redistribution |
| Images | Private / local only |

## Data collection

Images are sourced from the LAION-natural subset, a filtered version of LAION-2B containing images judged to be "natural" photographs. The pipeline applies multiple filtering stages:

1. **Caption scoring** — keyword and heuristic matching for fashion/outfit context
2. **CLIP image reranking** — scoring against person-in-outfit vs. product-only prompts
3. **Garment detection** — DETR-based detection to confirm visible clothing

No images are downloaded from the internet by this pipeline. All images come from pre-existing LAION-natural tar shards on the compute server.

## Known biases

- **Geographic/cultural:** LAION data is predominantly Western-centric. Fashion norms from non-Western cultures are underrepresented.
- **Body diversity:** Over-representation of certain body types, ages, and presentations due to web image distribution.
- **Photography quality:** Professional photographs are over-represented relative to everyday outfit photos.
- **Gender presentation:** Likely skewed toward stereotypical gender presentations.
- **Class markers:** "Polished" or "expensive-looking" axes may conflate clothing quality with photography quality and socioeconomic markers.

## Content safety

- Caption filtering excludes children, minors, NSFW content, swimwear, and explicit material via keyword matching.
- No automated NSFW classifier or age detector has been applied.
- No face detection or blur has been applied.
- Images should be treated as potentially containing identifiable individuals.

## Intended use

- Research into foundation-model representations of clothing and style
- Portfolio demonstration of data curation and embedding analysis
- Internal exploration and development

## Not intended for

- Public redistribution
- Fashion recommendation to real users
- Aesthetic or style judgment of individuals
- Any use involving minors
- Commercial use without licensing review

## Maintenance

This is a snapshot dataset for a one-time project. It is not actively maintained or updated.
