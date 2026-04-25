"""Caption-based filtering for fashion/outfit images.

This is a debug bootstrap filter, not a safety classifier.  It uses keyword
matching to bias toward captions that describe people wearing clothes and away
from industrial, product-only, or irrelevant contexts.

The filter uses a two-tier approach:

1. **Context terms** (e.g. "outfit", "wearing", "street style") strongly imply
   a person-in-clothing context on their own.
2. **Garment terms** (e.g. "jacket", "coat", "dress") are ambiguous — they
   appear in product listings, industrial docs, and non-fashion contexts.
   A garment term only counts if the caption also contains a **person hint**
   (e.g. "woman", "man", "model", "wearing").

Both tiers are gated by an exclusion list that rejects industrial, medical,
product-only, and other non-fashion contexts.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class RejectReason(str, Enum):
    EMPTY = "empty_caption"
    NO_FASHION_SIGNAL = "no_fashion_signal"
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


# ---------------------------------------------------------------------------
# Term lists
# ---------------------------------------------------------------------------

# Context terms: strongly imply person + clothing on their own.
# Note: bare "fashion" is intentionally excluded — it matches product names
# ("Fashion Heart Necklace"), decor, and jewelry organizers.  Only compound
# forms that imply a person/outfit context are kept.
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

# Garment terms: only count when a person hint is also present.
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

# Person hints: at least one must appear alongside a garment term.
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

# Exclusion terms: reject regardless of fashion signal.
EXCLUSION_TERMS = (
    # People / safety
    "baby",
    "infant",
    "toddler",
    "child",
    "children",
    "kid ",
    "kids",
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
    # Home decor / supplies / furniture
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
    # News / events / politics / protest
    "protest",
    "protester",
    "demonstrat",  # demonstration, demonstrators
    "rally",
    "riot",
    "police",
    "officer",
    "arrested",
    "arrest",
    "injured",
    "injur",  # injuries
    "casualt",  # casualty, casualties
    "shooting",
    "bombing",
    "explosion",
    "fire department",
    "firefight",
    "rescue",
    "disaster",
    "flood",
    "earthquake",
    "hurricane",
    "tornado",
    "wildfire",
    "evacuat",  # evacuation, evacuated
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
    "televisi",  # television
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
    # Animals
    "dog ",
    "cat ",
    "horse",
    "pet ",
    "puppy",
    "kitten",
    "animal",
)

# Backward-compatible flat lists (kept for any external consumers)
FASHION_TERMS = CONTEXT_TERMS + GARMENT_TERMS


def filter_caption(caption: str | None) -> FilterResult:
    """Evaluate whether *caption* describes a person wearing clothing.

    Returns a :class:`FilterResult` with accept/reject status and reason.
    """
    if not caption:
        return FilterResult(accepted=False, reason=RejectReason.EMPTY)

    text = caption.lower()

    # Check exclusions first
    for term in EXCLUSION_TERMS:
        if term in text:
            return FilterResult(accepted=False, reason=RejectReason.EXCLUDED, matched_term=term)

    # Tier 1: context terms are sufficient on their own
    for term in CONTEXT_TERMS:
        if term in text:
            return FilterResult(accepted=True, matched_term=term)

    # Tier 2: garment terms require a person hint
    has_person = any(hint in text for hint in PERSON_HINTS)
    if has_person:
        for term in GARMENT_TERMS:
            if term in text:
                return FilterResult(accepted=True, matched_term=term)

    return FilterResult(accepted=False, reason=RejectReason.NO_FASHION_SIGNAL)


def caption_matches_fashion(caption: str | None) -> bool:
    """Simple boolean interface for backward compatibility."""
    return filter_caption(caption).accepted
