"""LLM-based treatment explanation layer.

Takes the structured :class:`~src.rag_engine.TreatmentPlan` (precise,
template-driven) and asks Gemini to render it as warm, farmer-friendly
prose in the requested language — the layer between "correct data" and
"something a farmer actually enjoys reading."
"""

from __future__ import annotations

from dataclasses import dataclass

from src.config import configure_logging
from src.rag_engine import TreatmentPlan

logger = configure_logging(__name__)

SYSTEM_INSTRUCTION = (
    "You are an agricultural extension advisor writing for Indian smallholder "
    "farmers. Explain treatment plans in short, simple sentences. Never "
    "invent facts, dosages, or chemical names beyond what is given in the "
    "structured data provided. If information is marked 'N/A' or missing, "
    "say so plainly rather than guessing. Keep the tone reassuring but "
    "direct about urgency when the disease is severe."
)


class LLMAdvisorError(Exception):
    """Raised when the LLM explanation cannot be generated."""


@dataclass
class ExplainedTreatment:
    """Farmer-facing explanation generated from a structured treatment plan."""

    language: str
    explanation_text: str
    source_plan: TreatmentPlan

    def to_dict(self) -> dict:
        return {
            "language": self.language,
            "explanation_text": self.explanation_text,
            "disease_name_en": self.source_plan.disease_name_en,
            "disease_name_hi": self.source_plan.disease_name_hi,
        }


def build_explanation_prompt(plan: TreatmentPlan, language: str = "hinglish") -> str:
    """Build the Gemini prompt for explaining a treatment plan.

    Args:
        plan: Structured treatment plan from :class:`~src.rag_engine.RAGEngine`.
        language: ``"hindi"``, ``"english"``, or ``"hinglish"``.

    Returns:
        A complete prompt string.
    """
    language_instruction = {
        "hindi": "Respond entirely in Hindi (Devanagari script).",
        "english": "Respond entirely in English.",
        "hinglish": "Respond in Hinglish (Hindi content, Latin/Roman script) — this is how most farmers text on WhatsApp.",
    }.get(language, "Respond in Hinglish (Hindi content, Latin/Roman script).")

    chem = plan.chemical_treatment or {}
    return (
        f"{SYSTEM_INSTRUCTION}\n\n{language_instruction}\n\n"
        f"Structured treatment data:\n"
        f"- Disease: {plan.disease_name_en} ({plan.disease_name_hi})\n"
        f"- Immediate action: {plan.immediate_action_en}\n"
        f"- Chemical treatment: {chem.get('pesticide_name', 'N/A')} "
        f"(brands: {chem.get('brand_examples', 'N/A')}), "
        f"dose: {chem.get('dosage', 'N/A')}, frequency: {chem.get('frequency', 'N/A')}\n"
        f"- Safety: {chem.get('safety_en', 'N/A')}\n"
        f"- Organic alternative: {plan.organic_alternative_en}\n"
        f"- Prevention: {plan.prevention_en}\n"
        f"- Estimated recovery: {plan.estimated_recovery}\n\n"
        f"Write a short (4-6 sentence) farmer-friendly explanation covering what "
        f"to do right now, what to spray and how, and one prevention tip for "
        f"next season. Do not add any information not present above."
    )


def explain_treatment_plan(
    plan: TreatmentPlan, language: str = "hinglish", gemini_client=None
) -> ExplainedTreatment:
    """Generate a farmer-friendly explanation of a structured treatment plan.

    Args:
        plan: Output of ``RAGEngine.get_treatment_plan``.
        language: ``"hindi"``, ``"english"``, or ``"hinglish"``.
        gemini_client: Optional pre-configured model instance (for testing);
            defaults to a fresh ``gemini-1.5-flash`` model using
            ``GEMINI_API_KEY`` from the environment.

    Returns:
        An :class:`ExplainedTreatment`.

    Raises:
        LLMAdvisorError: If the Gemini call fails or returns empty output.
    """
    if gemini_client is None:
        from src.llm_client import GeminiClientError, GeminiTextClient

        try:
            gemini_client = GeminiTextClient()
        except GeminiClientError as exc:
            raise LLMAdvisorError(str(exc)) from exc

    prompt = build_explanation_prompt(plan, language)

    try:
        response = gemini_client.generate_content(prompt)
    except Exception as exc:  # noqa: BLE001
        raise LLMAdvisorError(f"Gemini explanation call failed: {exc}") from exc

    explanation_text = (getattr(response, "text", "") or "").strip()
    if not explanation_text:
        raise LLMAdvisorError("Gemini returned an empty explanation.")

    logger.info("Generated %s explanation for '%s'.", language, plan.disease_name_en)
    return ExplainedTreatment(language=language, explanation_text=explanation_text, source_plan=plan)


def build_fallback_explanation(plan: TreatmentPlan, language: str = "english") -> ExplainedTreatment:
    """Deterministic, non-LLM fallback explanation.

    Used when ``GEMINI_API_KEY`` is missing or the API call fails, so the
    app degrades to plain structured text instead of showing an error.

    Args:
        plan: Structured treatment plan.
        language: ``"hindi"`` or ``"english"`` (Hinglish falls back to English).

    Returns:
        An :class:`ExplainedTreatment` built from string templates, no LLM call.
    """
    chem = plan.chemical_treatment or {}
    if language == "hindi":
        text = (
            f"{plan.disease_name_hi or plan.disease_name_en} पाया गया।\n"
            f"तुरंत करें: {plan.immediate_action_hi or plan.immediate_action_en}\n"
            f"दवाई: {chem.get('pesticide_name', 'N/A')} — {chem.get('dosage', 'N/A')}, "
            f"{chem.get('frequency', 'N/A')}\n"
            f"सुरक्षा: {chem.get('safety_hi', 'N/A')}\n"
            f"जैविक विकल्प: {plan.organic_alternative_hi or plan.organic_alternative_en}\n"
            f"रोकथाम: {plan.prevention_hi or plan.prevention_en}"
        )
    else:
        text = (
            f"Detected: {plan.disease_name_en}.\n"
            f"Immediate action: {plan.immediate_action_en}\n"
            f"Chemical treatment: {chem.get('pesticide_name', 'N/A')} — "
            f"{chem.get('dosage', 'N/A')}, {chem.get('frequency', 'N/A')}\n"
            f"Safety: {chem.get('safety_en', 'N/A')}\n"
            f"Organic alternative: {plan.organic_alternative_en}\n"
            f"Prevention: {plan.prevention_en}"
        )
    return ExplainedTreatment(language=language, explanation_text=text, source_plan=plan)
