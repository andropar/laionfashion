from __future__ import annotations


FASHION_TERMS = (
    "outfit",
    "fashion",
    "street style",
    "wearing",
    "dressed",
    "dress",
    "shirt",
    "jacket",
    "coat",
    "sweater",
    "pants",
    "trousers",
    "jeans",
    "skirt",
    "shoes",
    "boots",
    "sneakers",
)

EXCLUSION_TERMS = (
    "baby",
    "child",
    "children",
    "kid",
    "kids",
    "underwear",
    "lingerie",
    "bikini",
    "swimwear",
    "nude",
    "nudity",
    "sexy",
    "porn",
    "cartoon",
    "drawing",
    "illustration",
    "logo",
    "text document",
)


def caption_matches_fashion(caption: str | None) -> bool:
    if not caption:
        return False
    text = caption.lower()
    return any(term in text for term in FASHION_TERMS) and not any(
        term in text for term in EXCLUSION_TERMS
    )

