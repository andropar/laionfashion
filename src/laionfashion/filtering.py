"""Caption-based filtering for fashion/outfit images.

This is a debug bootstrap filter, not a safety classifier.  It uses keyword
matching to bias toward captions that describe people wearing clothes and away
from industrial, product-only, or irrelevant contexts.

Two interfaces:

- **filter_caption()** — binary accept/reject with reject reason (backward
  compatible, used by ``broad`` and ``strict`` selection modes).
- **score_caption()** — numeric score with breakdown, used by the ``outfit``
  selection mode to prioritize visible-person-in-clothing captions over
  product-only garment captions.

Both are gated by the same exclusion list.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class RejectReason(str, Enum):
    EMPTY = "empty_caption"
    NO_FASHION_SIGNAL = "no_fashion_signal"
    NO_PERSON_CONTEXT = "no_person_context"
    BELOW_SCORE_THRESHOLD = "below_score_threshold"
    EXCLUDED = "excluded_term"


@dataclass(frozen=True)
class FilterResult:
    """Result of caption filtering with optional reject reason."""

    accepted: bool
    reason: RejectReason | None = None
    matched_term: str | None = None

    @property
    def rejected(self) -> bool:
        return not self.accepted


@dataclass(frozen=True)
class CaptionScore:
    """Numeric score for a caption with signal breakdown."""

    score: float
    signals: dict[str, float] = field(default_factory=dict)
    excluded: bool = False
    excluded_term: str | None = None

    @property
    def is_excluded(self) -> bool:
        return self.excluded


# ---------------------------------------------------------------------------
# Term lists
# ---------------------------------------------------------------------------

CONTEXT_TERMS = (
    "outfit",
    "street style",
    "streetwear",
    "wearing",
    "dressed in",
    "dressed up",
    "ootd",
    "lookbook",
    "fashion week",
    "fashion show",
    "fashion photo",
    "fashion blog",
    "fashion model",
    "fashion style",
    "fashion look",
    "street fashion",
    "style outfit",
    "runway",
)

GARMENT_TERMS = (
    "dress",
    "shirt",
    "blouse",
    "jacket",
    "blazer",
    "coat",
    "overcoat",
    "sweater",
    "hoodie",
    "cardigan",
    "pants",
    "trousers",
    "jeans",
    "skirt",
    "shorts",
    "shoes",
    "boots",
    "sneakers",
    "heels",
    "sandals",
    "suit",
    "tuxedo",
    "gown",
    "romper",
    "jumpsuit",
    "vest",
    "scarf",
    "hat",
)

PERSON_HINTS = (
    "woman",
    "women",
    "man ",  # trailing space to avoid "manual", "many", "mannequin", etc.
    "men ",
    "lady",
    "girl",
    "boy",
    "model",
    "stylish",
    "posing",
    "portrait",
    "wearing",
    "wears",
    "wore",
    "dressed",
    "casual",
    "elegant",
    "chic",
)

EXCLUSION_TERMS = (
    # People / safety
    "baby",
    "infant",
    "toddler",
    "child",
    "children",
    "kid'",  # kid's, kids'
    "kid ",
    "kids",
    " son ",
    "daughter",
    # Explicit / inappropriate
    "underwear",
    "lingerie",
    "bikini",
    "swimwear",
    "swimsuit",
    "nude",
    "nudity",
    "naked",
    "sexy",
    "porn",
    "erotic",
    "nsfw",
    # Non-photo
    "cartoon",
    "drawing",
    "illustration",
    "anime",
    "manga",
    "sketch",
    "clipart",
    "vector",
    "logo",
    "icon",
    "emoji",
    "text document",
    "screenshot",
    "infographic",
    # Toys / plush / dolls
    "plush",
    "stuffed animal",
    "stuffed toy",
    "teddy bear",
    "doll",
    "figurine",
    "action figure",
    "toy ",
    "toys",
    "tutu",
    # Industrial / chemical / technical
    "conformal",
    "coating",
    "chemical",
    "seizure",
    "circuit",
    "pcb",
    "solder",
    "wiring",
    "harness",
    "cable",
    "connector",
    "gasket",
    "valve",
    "pipe",
    "tubing",
    "hydraulic",
    "pneumatic",
    "insulation",
    "resin",
    "epoxy",
    "adhesive",
    "sealant",
    "lubricant",
    "corrosion",
    "abrasive",
    "tensile",
    "polymer",
    "substrate",
    "electrode",
    "catalyst",
    "reagent",
    "solvent",
    "specimen",
    "microscope",
    "laboratory",
    # Medical / hospital
    "surgical",
    "medical",
    "clinical",
    "patient",
    "diagnosis",
    "symptom",
    "therapy",
    "prosthetic",
    "orthopedic",
    "pharmaceutical",
    "hospital",
    "nurse",
    "doctor",
    "surgery",
    "icu",
    "emergency room",
    "ambulance",
    # Product-only / e-commerce without person
    "white background isolated",
    "product photo on white",
    "flat lay",
    "mannequin",
    # Jewelry / accessories without clothing context
    "necklace",
    "bracelet",
    "earring",
    "pendant",
    "brooch",
    "anklet",
    "jewelry",
    "jewellery",
    # Home decor / supplies / furniture / renovation
    "decoration",
    "decor",
    "supplies",
    "ornament",
    "centerpiece",
    "dining table",
    "christmas decoration",
    "furniture",
    "upholstery",
    "curtain",
    "mattress",
    "sofa",
    "couch",
    "organizer",
    "storage",
    "shelf",
    "cabinet",
    "pillow",
    "cushion",
    "tablecloth",
    "placemat",
    "remodel",
    "renovati",
    "camper",
    "caravan",
    # Automotive / mechanical
    "engine",
    "motor",
    "transmission",
    "exhaust",
    "brake",
    "suspension",
    "chassis",
    "bumper",
    "fender",
    "radiator",
    # Food / cooking
    "recipe",
    "ingredient",
    "baking",
    "cooking",
    "roast",
    # Ceremony / royalty / awards
    "ceremony",
    "coronation",
    "royal family",
    "queen elizabeth",
    "king charles",
    "prince ",
    "princess ",
    "award show",
    "red carpet",
    "oscar",
    "grammy",
    "emmy",
    "golden globe",
    # Theater / performance / fictional characters
    "theater",
    "theatre",
    "broadway",
    "ballet",
    "opera",
    "costume party",
    "cosplay",
    "halloween",
    "full monty",
    "superman",
    "superhero",
    "superheroine",
    # News / events / politics / protest
    "protest",
    "protester",
    "demonstrat",
    "rally",
    "riot",
    "police",
    "officer",
    "arrested",
    "arrest",
    "injured",
    "injur",
    "casualt",
    "shooting",
    "bombing",
    "explosion",
    "fire department",
    "firefight",
    "fire hat",
    "fire truck",
    "rescue",
    "disaster",
    "flood",
    "earthquake",
    "hurricane",
    "tornado",
    "wildfire",
    "evacuat",
    "election",
    "politician",
    "president",
    "senator",
    "congress",
    "parliament",
    "campaign",
    "democrat",
    "republican",
    "obama",
    "trump",
    "biden",
    "vote",
    "voter",
    "ballot",
    "televisi",
    "broadcast",
    "newscast",
    "reporter",
    "journalist",
    "press conference",
    "courtroom",
    "verdict",
    "trial",
    "lawsuit",
    "inmate",
    "prison",
    "jail",
    "refugee",
    "migrant",
    "military",
    "soldier",
    "troop",
    "combat",
    "warfare",
    "weapon",
    "tank ",
    "missile",
    "grenade",
    # Sports (non-fashion)
    "soccer",
    "football",
    "basketball",
    "baseball",
    "hockey",
    "tennis",
    "cricket",
    "rugby",
    "wrestling",
    "boxing",
    "stadium",
    "goalkeeper",
    "touchdown",
    "championship",
    "tournament",
    "lindelof",
    "bailly",
    # Animals
    "dog ",
    "cat ",
    "horse",
    "pet ",
    "puppy",
    "kitten",
    "animal",
)

# Backward-compatible flat lists
FASHION_TERMS = CONTEXT_TERMS + GARMENT_TERMS

# ---------------------------------------------------------------------------
# Scoring — weighted signal terms
# ---------------------------------------------------------------------------

# Strong outfit context: captions with these very likely show a person
# in visible clothing.  Score: +3.0 each (only first match counted).
_STRONG_OUTFIT_TERMS = (
    "outfit",
    "ootd",
    "what i wore",
    "fashion week",
    "fashion show",
    "runway",
    "street style",
    "street fashion",
    "lookbook",
    "fashion model",
)

# Medium context: fashion-adjacent but slightly weaker signal.  +2.0.
_MEDIUM_CONTEXT_TERMS = (
    "fashion photo",
    "fashion blog",
    "fashion style",
    "fashion look",
    "style outfit",
    "streetwear",
    "dressed in",
    "dressed up",
)

# Penalty terms: suggest non-fashion or misleading context.  -1.5 each.
_PENALTY_TERMS = (
    "half-dressed",
    "half dressed",
    "undress",
    "search result",
    "google",
    "catalog",
    "catalogue",
    "buy now",
    "shop now",
    "add to cart",
    "free shipping",
    "price",
    "discount",
    "coupon",
    "wholesale",
    "bulk order",
    "amazon",
    "ebay",
    "aliexpress",
    "alibaba",
)

# Selection mode thresholds
SELECTION_MODES: dict[str, float] = {
    "broad": 0.5,
    "strict": 1.5,
    "outfit": 2.5,
}


def score_caption(caption: str | None) -> CaptionScore:
    """Score a caption for outfit/person-in-clothing relevance.

    Returns a :class:`CaptionScore` with a numeric score and signal breakdown.
    Excluded captions get ``score = -inf``.

    Score components:
    - Strong outfit terms: +3.0 (first match)
    - Medium context terms: +2.0 (first match)
    - "wearing" context: +1.5
    - Garment term present: +1.0
    - Person hint present: +1.0
    - Garment + person combo bonus: +0.5
    - Penalty terms: -1.5 each
    """
    if not caption:
        return CaptionScore(score=-float("inf"), signals={"empty": -float("inf")})

    text = caption.lower()

    # Check exclusions
    for term in EXCLUSION_TERMS:
        if term in text:
            return CaptionScore(
                score=-float("inf"),
                excluded=True,
                excluded_term=term,
                signals={"excluded": -float("inf")},
            )

    signals: dict[str, float] = {}
    score = 0.0

    # Strong outfit context
    for term in _STRONG_OUTFIT_TERMS:
        if term in text:
            signals["strong_outfit"] = 3.0
            score += 3.0
            break

    # Medium context (only if no strong match)
    if "strong_outfit" not in signals:
        for term in _MEDIUM_CONTEXT_TERMS:
            if term in text:
                signals["medium_context"] = 2.0
                score += 2.0
                break

    # "wearing" as context (only if no strong/medium match yet)
    if "strong_outfit" not in signals and "medium_context" not in signals:
        if "wearing" in text:
            signals["wearing_context"] = 1.5
            score += 1.5

    # Garment term present
    has_garment = any(term in text for term in GARMENT_TERMS)
    if has_garment:
        signals["garment_term"] = 1.0
        score += 1.0

    # Person hint present
    has_person = any(hint in text for hint in PERSON_HINTS)
    if has_person:
        signals["person_hint"] = 1.0
        score += 1.0

    # Combo bonus: garment + person
    if has_garment and has_person:
        signals["garment_person_combo"] = 0.5
        score += 0.5

    # Penalty terms
    penalty = 0.0
    for term in _PENALTY_TERMS:
        if term in text:
            penalty -= 1.5
    if penalty < 0:
        signals["penalty"] = penalty
        score += penalty

    return CaptionScore(score=score, signals=signals)


# ---------------------------------------------------------------------------
# Binary filter (backward compatible)
# ---------------------------------------------------------------------------


def filter_caption(
    caption: str | None,
    *,
    require_person_context: bool = False,
    min_score: float | None = None,
) -> FilterResult:
    """Evaluate whether *caption* describes a person wearing clothing.

    Parameters
    ----------
    caption:
        The image caption to evaluate.
    require_person_context:
        When *True*, accepted captions must contain at least one person hint.
    min_score:
        When set, use score-based filtering instead of tier logic.
        Captions scoring below this threshold are rejected.

    Returns a :class:`FilterResult` with accept/reject status and reason.
    """
    if not caption:
        return FilterResult(accepted=False, reason=RejectReason.EMPTY)

    text = caption.lower()

    # Check exclusions first
    for term in EXCLUSION_TERMS:
        if term in text:
            return FilterResult(accepted=False, reason=RejectReason.EXCLUDED, matched_term=term)

    # Score-based mode
    if min_score is not None:
        sc = score_caption(caption)
        if sc.score >= min_score:
            return FilterResult(accepted=True, matched_term=f"score={sc.score:.1f}")
        return FilterResult(
            accepted=False,
            reason=RejectReason.BELOW_SCORE_THRESHOLD,
            matched_term=f"score={sc.score:.1f}<{min_score}",
        )

    # Tier-based mode (original logic)
    has_person = any(hint in text for hint in PERSON_HINTS)

    for term in CONTEXT_TERMS:
        if term in text:
            if require_person_context and not has_person:
                return FilterResult(
                    accepted=False,
                    reason=RejectReason.NO_PERSON_CONTEXT,
                    matched_term=term,
                )
            return FilterResult(accepted=True, matched_term=term)

    if has_person:
        for term in GARMENT_TERMS:
            if term in text:
                return FilterResult(accepted=True, matched_term=term)

    return FilterResult(accepted=False, reason=RejectReason.NO_FASHION_SIGNAL)


def caption_matches_fashion(
    caption: str | None,
    *,
    require_person_context: bool = False,
    min_score: float | None = None,
) -> bool:
    """Simple boolean interface for backward compatibility."""
    return filter_caption(
        caption, require_person_context=require_person_context, min_score=min_score
    ).accepted
