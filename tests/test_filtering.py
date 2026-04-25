"""Tests for laionfashion.filtering – caption filter logic and known false positives."""

from __future__ import annotations

import pytest

from laionfashion.filtering import (
    CaptionScore,
    FilterResult,
    RejectReason,
    SELECTION_MODES,
    caption_matches_fashion,
    filter_caption,
    score_caption,
)


# ---------------------------------------------------------------------------
# True positives: captions that should be accepted
# ---------------------------------------------------------------------------


class TestAccepts:
    @pytest.mark.parametrize(
        "caption",
        [
            "A woman wearing a red dress on the street",
            "Street style fashion photo from Milan",
            "Man dressed in a suit and tie at a wedding",
            "Casual outfit of the day: jeans and a hoodie",
            "Fashion week lookbook 2024",
            "Woman in a stylish jacket walking downtown",
            "Model posing in a black gown",
            "Photo of a person wearing sneakers and jeans",
            "She wore a casual sweater and skirt",
            "Elegant woman in heels and blazer",
            "OOTD minimalist style",
            "Portrait of a man wearing a coat in winter",
            "Picture of a lady in a floral dress at a garden party",
        ],
    )
    def test_accepts_fashion_captions(self, caption: str) -> None:
        result = filter_caption(caption)
        assert result.accepted, f"Should accept: {caption!r}, got reason={result.reason}"

    def test_backward_compat_bool(self) -> None:
        assert caption_matches_fashion("Woman wearing a red dress") is True
        assert caption_matches_fashion("conformal coating jacket") is False


# ---------------------------------------------------------------------------
# Known false positives from Raven that must now be rejected
# ---------------------------------------------------------------------------


class TestKnownFalsePositives:
    @pytest.mark.parametrize(
        "caption",
        [
            # Industrial / chemical (round 1)
            "conformal coating on circuit board jacket",
            "conformal coat applied to PCB assembly",
            "seizure jacket for pipe insulation",
            "chemical resistant jacket for industrial use",
            "waterproof coating for engine jacket",
            "heat shrink jacket for cable harness",
            # Product-only / no person context
            "jacket on white background isolated",
            "flat lay of dress shirt and pants",
            "mannequin wearing summer dress",
            # Automotive
            "engine jacket cooling system exhaust",
            "brake shoes replacement guide",
            # Medical
            "surgical coat for clinical use",
            "patient wearing medical jacket",
            # Food
            "potato jacket baking recipe",
            "cooking jacket recipe roast",
            # Animals
            "dog wearing a jacket in the park",
            "cat dress up costume for pet",
        ],
    )
    def test_rejects_round1_false_positives(self, caption: str) -> None:
        result = filter_caption(caption)
        assert result.rejected, f"Should reject: {caption!r}"

    @pytest.mark.parametrize(
        "caption",
        [
            # Exact Raven false positives (round 2)
            "Fashion Christmas Decoration Supplies For Dining Table",
            "A man watches a television showing Barack Obama giving a speech",
            "Masked and gowned person checking person in hospital bed",
            "Dozens of people, mostly police, were injured in the protest",
            "Fashion Heart Necklace",
            "InterDesign Classico Hanging Fashion for Jewelry Organizer",
            # Related patterns
            "Fashion bracelet gold plated pendant",
            "Fashion earring set for women accessories",
            "Fashion decor pillow cushion cover",
            "People arrested during protest rally downtown",
            "Person rescued from fire by firefighters",
            "Man watching football game on television at stadium",
            "Woman in hospital gown after surgery",
            "Soccer player wearing jersey and shorts",
            "Military soldier wearing combat boots",
        ],
    )
    def test_rejects_round2_false_positives(self, caption: str) -> None:
        result = filter_caption(caption)
        assert result.rejected, f"Should reject: {caption!r}"

    @pytest.mark.parametrize(
        "caption",
        [
            # Exact Raven false positives (round 3)
            "Eva Amurri Martino's son Major wears a fire hat and boots",
            "The Queen wearing a blue dress at the coronation ceremony",
            "Plush toy wearing a tutu on display",
            "15 camper remodel ideas that will inspire you",
            "The stats that suggest Lindelof and Bailly are improving",
            "Half-dressed actors in a theater production",
            # Related patterns
            "Daughter wearing princess costume at Halloween party",
            "Teddy bear doll in a knitted sweater",
            "Prince William wearing suit at royal ceremony",
            "Oscar award show red carpet arrivals",
            "Ballet dancer in costume on broadway stage",
            "Cosplay outfit at convention center",
        ],
    )
    def test_rejects_round3_false_positives(self, caption: str) -> None:
        result = filter_caption(caption)
        assert result.rejected, f"Should reject: {caption!r}"


# ---------------------------------------------------------------------------
# True negatives: clearly non-fashion captions
# ---------------------------------------------------------------------------


class TestRejectsNonFashion:
    @pytest.mark.parametrize(
        "caption",
        [
            "A beautiful sunset over the mountains",
            "Server rack in a data center",
            "Diagram of cellular mitosis",
            "Annual revenue chart 2023",
            "",
            None,
        ],
    )
    def test_rejects_non_fashion(self, caption: str | None) -> None:
        result = filter_caption(caption)
        assert result.rejected


# ---------------------------------------------------------------------------
# Exclusion terms
# ---------------------------------------------------------------------------


class TestExclusions:
    @pytest.mark.parametrize(
        "caption",
        [
            "Baby wearing a cute outfit",
            "Children's fashion show",
            "Sexy lingerie collection",
            "Anime character wearing a school uniform",
            "Cartoon drawing of a person in a dress",
            "Swimsuit bikini collection",
        ],
    )
    def test_exclusions_override_fashion(self, caption: str) -> None:
        result = filter_caption(caption)
        assert result.rejected
        assert result.reason == RejectReason.EXCLUDED


# ---------------------------------------------------------------------------
# Two-tier logic
# ---------------------------------------------------------------------------


class TestTwoTierLogic:
    def test_context_term_sufficient_alone(self) -> None:
        """Context terms like 'outfit' don't need a person hint."""
        result = filter_caption("Best outfit for spring 2024")
        assert result.accepted

    def test_garment_term_alone_rejected(self) -> None:
        """Garment terms without a person hint are not enough."""
        result = filter_caption("Red jacket available in all sizes")
        assert result.rejected
        assert result.reason == RejectReason.NO_FASHION_SIGNAL

    def test_garment_term_with_person_hint_accepted(self) -> None:
        """Garment terms + person hint = accepted."""
        result = filter_caption("Woman in a red jacket at the cafe")
        assert result.accepted

    def test_garment_plus_style_hint(self) -> None:
        """'style' counts as a person hint."""
        result = filter_caption("Casual style brown jacket")
        assert result.accepted


# ---------------------------------------------------------------------------
# FilterResult details
# ---------------------------------------------------------------------------


class TestFilterResult:
    def test_empty_caption(self) -> None:
        result = filter_caption("")
        assert result.reason == RejectReason.EMPTY

    def test_none_caption(self) -> None:
        result = filter_caption(None)
        assert result.reason == RejectReason.EMPTY

    def test_excluded_reports_term(self) -> None:
        result = filter_caption("conformal coating jacket")
        assert result.reason == RejectReason.EXCLUDED
        assert result.matched_term == "conformal"

    def test_accepted_reports_matched_term(self) -> None:
        result = filter_caption("Street style photo at fashion week")
        assert result.accepted
        assert result.matched_term is not None

    def test_no_signal_reason(self) -> None:
        result = filter_caption("A blue thing sitting on a bench")
        assert result.reason == RejectReason.NO_FASHION_SIGNAL


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestRequirePersonContext:
    def test_strict_rejects_context_without_person(self) -> None:
        """In strict mode, 'outfit' alone is not enough without a person hint."""
        result = filter_caption("Best outfit for spring 2024", require_person_context=True)
        assert result.rejected
        assert result.reason == RejectReason.NO_PERSON_CONTEXT

    def test_strict_accepts_context_with_person(self) -> None:
        result = filter_caption("Woman shows her outfit for spring", require_person_context=True)
        assert result.accepted

    def test_strict_accepts_wearing_plus_garment(self) -> None:
        """'wearing' serves as both context term and person hint."""
        result = filter_caption("She is wearing a red dress", require_person_context=True)
        assert result.accepted

    def test_default_mode_accepts_context_alone(self) -> None:
        result = filter_caption("Best outfit for spring 2024", require_person_context=False)
        assert result.accepted

    def test_backward_compat_bool_strict(self) -> None:
        assert caption_matches_fashion("Best outfit for spring", require_person_context=True) is False
        assert caption_matches_fashion("Woman in a great outfit", require_person_context=True) is True


class TestEdgeCases:
    def test_case_insensitive(self) -> None:
        assert filter_caption("WOMAN WEARING A DRESS").accepted
        assert filter_caption("OUTFIT of the day").accepted

    def test_partial_word_garment_with_exclusion(self) -> None:
        """'coat' in 'coating' should be caught by 'coating' exclusion."""
        result = filter_caption("coating process for jacket")
        assert result.rejected

    def test_dress_as_verb_with_person(self) -> None:
        """'dressed in' is a context term."""
        result = filter_caption("She dressed in layers for winter")
        assert result.accepted


# ---------------------------------------------------------------------------
# Round 4 false positives
# ---------------------------------------------------------------------------


class TestRound4FalsePositives:
    @pytest.mark.parametrize(
        "caption",
        [
            "For people who want to see half-dressed men dancing... The Full Monty",
            "Not many know that Brandon Routh's Superman costume was redesigned",
            "Kid's TAO Short Sleeves White Dotted Stylish Casual Shirt",
        ],
    )
    def test_rejects_round4_false_positives(self, caption: str) -> None:
        result = filter_caption(caption)
        assert result.rejected, f"Should reject: {caption!r}"

    @pytest.mark.parametrize(
        "caption",
        [
            "Buy now women's dress free shipping discount",
            "Amazon best seller casual jacket wholesale price",
            "Add to cart men's suit ebay listing",
        ],
    )
    def test_penalty_captions_rejected_in_outfit_mode(self, caption: str) -> None:
        """E-commerce spam captions should score below outfit threshold."""
        sc = score_caption(caption)
        assert sc.score < SELECTION_MODES["outfit"]


# ---------------------------------------------------------------------------
# Caption scoring
# ---------------------------------------------------------------------------


class TestScoreCaption:
    def test_empty_caption(self) -> None:
        sc = score_caption("")
        assert sc.score == -float("inf")

    def test_excluded_caption(self) -> None:
        sc = score_caption("conformal coating jacket")
        assert sc.is_excluded
        assert sc.score == -float("inf")

    def test_strong_outfit_term(self) -> None:
        sc = score_caption("Great outfit for spring 2024")
        assert sc.signals.get("strong_outfit") == 3.0
        assert sc.score >= 3.0

    def test_medium_context_term(self) -> None:
        sc = score_caption("Fashion blog post about new trends")
        assert sc.signals.get("medium_context") == 2.0

    def test_wearing_context(self) -> None:
        sc = score_caption("A beautiful mural on the wall wearing ivy")
        assert sc.signals.get("wearing_context") == 1.5

    def test_garment_plus_person(self) -> None:
        sc = score_caption("Woman in a red jacket at the cafe")
        assert sc.signals.get("garment_term") == 1.0
        assert sc.signals.get("person_hint") == 1.0
        assert sc.signals.get("garment_person_combo") == 0.5
        assert sc.score >= 2.5

    def test_garment_only(self) -> None:
        """Garment without person hint should score low."""
        sc = score_caption("Red jacket available in all sizes")
        assert sc.signals.get("garment_term") == 1.0
        assert "person_hint" not in sc.signals
        assert sc.score < SELECTION_MODES["strict"]

    def test_penalty_reduces_score(self) -> None:
        sc = score_caption("Buy now women's dress free shipping discount")
        assert sc.signals.get("penalty", 0) < 0
        # Should score lower than a clean version
        sc_clean = score_caption("Women's dress for evening event")
        assert sc.score < sc_clean.score

    def test_strong_outfit_beats_product(self) -> None:
        """Outfit term should score higher than product-only garment."""
        sc_outfit = score_caption("My daily outfit: jeans and a tee")
        sc_product = score_caption("Jeans denim blue size 32")
        assert sc_outfit.score > sc_product.score

    def test_outfit_mode_threshold(self) -> None:
        """Strong outfit captions pass outfit mode, weak ones don't."""
        strong = "Street style outfit at fashion week with model posing"
        weak = "Blue dress on rack"
        assert score_caption(strong).score >= SELECTION_MODES["outfit"]
        assert score_caption(weak).score < SELECTION_MODES["outfit"]


# ---------------------------------------------------------------------------
# Score-based filtering (min_score parameter)
# ---------------------------------------------------------------------------


class TestScoreBasedFiltering:
    def test_min_score_accepts_high(self) -> None:
        result = filter_caption("Woman wearing a great outfit today", min_score=2.0)
        assert result.accepted

    def test_min_score_rejects_low(self) -> None:
        result = filter_caption("Red jacket available in all sizes", min_score=2.0)
        assert result.rejected
        assert result.reason == RejectReason.BELOW_SCORE_THRESHOLD

    def test_min_score_still_excludes(self) -> None:
        result = filter_caption("conformal coating jacket", min_score=0.0)
        assert result.rejected
        assert result.reason == RejectReason.EXCLUDED

    def test_selection_modes_exist(self) -> None:
        assert "broad" in SELECTION_MODES
        assert "strict" in SELECTION_MODES
        assert "outfit" in SELECTION_MODES
        assert SELECTION_MODES["broad"] < SELECTION_MODES["strict"] < SELECTION_MODES["outfit"]
