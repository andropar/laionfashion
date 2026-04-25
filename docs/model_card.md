# Model card

## Models used

This project does not train new models. It uses the following pretrained foundation models:

### OpenCLIP ViT-B/32 (primary)

| Field | Value |
|-------|-------|
| Architecture | Vision Transformer B/32 |
| Source | `open_clip` / `laion400m_e31` (fallback: `openai`) |
| Purpose | Image and text embedding, prompt-direction axes, outfit scoring |
| Embedding dim | 512 |
| License | MIT (open_clip), model weights vary by pretrained source |

Used for:
- Full-image embeddings
- Garment crop embeddings
- Person-in-outfit vs. product-only image scoring
- Prompt-direction style axis computation

### Marqo FashionCLIP (comparison)

| Field | Value |
|-------|-------|
| Architecture | ViT-B/32 fine-tuned on fashion data |
| Source | `hf-hub:Marqo/marqo-fashionCLIP` |
| Purpose | Fashion-specific embedding comparison |
| Embedding dim | 512 |
| License | See Marqo/marqo-fashionCLIP on HuggingFace |

Used for:
- Comparative embedding analysis
- Fashion-specific retrieval evaluation

### Fashion DETR (garment detection)

| Field | Value |
|-------|-------|
| Architecture | Conditional DETR (ResNet-50) |
| Source | `yainage90/fashion-object-detection` on HuggingFace |
| Purpose | Garment bounding box detection |
| Categories | top, bottom, dress, outer, shoes, hat, bag |
| mAP | 0.7542 |
| License | See model card on HuggingFace |

Used for:
- Detecting individual garments in outfit images
- Producing garment crops for per-garment embedding

## Limitations

- **CLIP encodes photography, not just clothing.** Lighting, background, pose, and image quality strongly influence embeddings. A professional photo of casual clothes may embed closer to other professional photos than to similar casual clothes in amateur photos.
- **FashionCLIP improves item similarity but may not improve style/compatibility axes.** It was trained for product matching, not outfit-level understanding.
- **Prompt-direction axes are approximate.** The difference between two text embeddings is a crude proxy for a style dimension. Some axes (formal/casual) may be less discriminative than others (colorful/neutral).
- **DETR misses garments in complex scenes.** Low-resolution source images, occlusion, unusual poses, and non-standard garments reduce detection quality.

## Ethical considerations

- Models were trained on web-scale data with known biases.
- Style/aesthetic judgments are model-derived, not objective.
- No attempt is made to rank individuals or their appearance.
- See `docs/data_card.md` for data-specific concerns.
