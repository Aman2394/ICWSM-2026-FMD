"""
Perturbation prompt templates for data augmentation.
Each key is a perturbation type; value is the instruction given to the LLM.
"""

PERTURBATION_PROMPTS: dict[str, str] = {
    "misleading_framing": (
        "Rewrite using alarming/negative word choice, but change NO facts."
    ),
    "cherry_picked_context": (
        "Add one true-sounding sentence from a different time period/sector "
        "that misleads."
    ),
    "false_attribution": (
        "Change who said or caused something. Keep the original claim unchanged."
    ),
    "correlation_as_causation": (
        "Rewrite to imply causation where the original only implies coincidence."
    ),
    "omission_bias": (
        "Remove one qualifying sentence (e.g., 'however...') that changes "
        "the overall meaning."
    ),
    "scale_distortion": (
        "Change magnitude words: 'some'→'most', 'one analyst'→'analysts'."
    ),
    "hedge_manipulation": (
        "Flip certainty: 'will increase'→'might increase' or vice versa."
    ),
    "detail_hallucination": (
        "Add one specific but fabricated detail: a fake quote, invented metric, "
        "or made-up comparison."
    ),
    "composite": (
        "Apply two subtle changes simultaneously "
        "(e.g., a numerical tweak + a slight sentiment shift)."
    ),
}


def build_prompt(paragraph: str, perturbation_type: str) -> str:
    """Build a full LLM prompt for a given perturbation type."""
    if perturbation_type not in PERTURBATION_PROMPTS:
        raise ValueError(
            f"Unknown perturbation type '{perturbation_type}'. "
            f"Choose from: {list(PERTURBATION_PROMPTS)}"
        )
    instruction = PERTURBATION_PROMPTS[perturbation_type]
    return (
        f"You are a financial journalism editor.\n\n"
        f"TASK: {instruction}\n\n"
        f"ORIGINAL PARAGRAPH:\n{paragraph}\n\n"
        f"RULES:\n"
        f"- Output ONLY the rewritten paragraph, nothing else.\n"
        f"- Keep the paragraph length similar to the original.\n"
        f"- Make the change subtle — it should not be obvious to a casual reader.\n\n"
        f"REWRITTEN PARAGRAPH:"
    )
