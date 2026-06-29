"""Step 2 - Gemini report generator.

Takes the DR grade produced by the specialist vision model plus the fundus
image, and asks Gemini (a vision-language LLM) to write a structured, plain-
language clinical report AROUND that fixed grade. Gemini does not diagnose or
re-grade; the grade comes from the trained model.
"""
from __future__ import annotations

from google import genai

from src.config import GEMINI_API_KEY, GEMINI_MODEL, CLASS_NAMES

_client = None


def _get_client():
    global _client
    if _client is None:
        if not GEMINI_API_KEY:
            raise EnvironmentError(
                "GEMINI_API_KEY is not set. Copy .env.example to .env and add your "
                "free key from https://aistudio.google.com"
            )
        _client = genai.Client(api_key=GEMINI_API_KEY)
    return _client


_PROMPT = """You are a clinical assistant drafting an EDUCATIONAL report about a retinal fundus photograph.

A specialist automated diabetic-retinopathy grading model (a dedicated computer-vision model, not you) has classified this image as:

    Grade {grade}: {label}

Treat this grade as the established finding. Do NOT re-grade or contradict it.
Write a structured report with these four sections:

1. Clinical interpretation - what Grade {grade} ({label}) means for the patient.
2. Observable features - retinal features typically associated with this grade
   (e.g. microaneurysms, dot/blot haemorrhages, hard exudates, cotton-wool spots,
   neovascularisation), and note any consistent features visible in the image.
3. Recommended follow-up - a typical monitoring or referral timeline for this grade.
4. Important caveats - state clearly that this is automated AI grading for research
   and education only, is not a diagnosis, and must be reviewed by a qualified
   ophthalmologist.

Keep it concise, factual, and in plain clinical language."""


def generate_report(grade: int, image=None) -> str:
    """Generate a clinical report for a given DR grade (optionally with the image)."""
    client = _get_client()
    label = CLASS_NAMES.get(grade, str(grade))
    prompt = _PROMPT.format(grade=grade, label=label)

    contents = [prompt]
    if image is not None:
        contents.append(image)          # PIL image; google-genai accepts it directly

    response = client.models.generate_content(model=GEMINI_MODEL, contents=contents)
    return (getattr(response, "text", "") or "").strip()


if __name__ == "__main__":
    # Usage:  python -m src.report_generator <grade> [image_path]
    import sys
    from PIL import Image

    grade = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    img = Image.open(sys.argv[2]) if len(sys.argv) > 2 else None
    print(generate_report(grade, img))
