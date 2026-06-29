"""Gradio app - Diabetic Retinopathy Clinical Report Generator.

Pipeline: upload fundus image -> trained DR model grades severity (0-4) ->
Gemini drafts a structured clinical report around that grade.

Run locally:   python app.py
This file is also the HuggingFace Spaces entry point.
"""
from __future__ import annotations

import gradio as gr

from src.dr_grader import grade_fundus
from src.report_generator import generate_report
from src.config import CLASS_NAMES

DISCLAIMER = (
    "&#9888;&#65039; **Research/education demo only - not for clinical use.** "
    "The grade is produced by an automated model and the report is AI-generated; "
    "both must be reviewed by a qualified ophthalmologist."
)


def _per_class_from_ordinal(ordinal):
    """Convert CORN ordinal probabilities P(>0..>3) into a per-class distribution."""
    ge = [1.0] + list(ordinal)                 # P(y>=0..4)
    pc = []
    for k in range(5):
        nxt = ge[k + 1] if k + 1 < len(ge) else 0.0
        pc.append(max(0.0, ge[k] - nxt))
    total = sum(pc) or 1.0
    return {CLASS_NAMES[k]: pc[k] / total for k in range(5)}


def analyze(image):
    if image is None:
        return "Please upload a fundus image.", None, {}, ""

    result = grade_fundus(image)
    grade, label = result["grade"], result["label"]
    grade_md = f"## Grade {grade}: {label}"
    confidences = _per_class_from_ordinal(result["ordinal_probabilities"])

    try:
        report = generate_report(grade, result["processed_image"])
    except Exception as exc:  # noqa: BLE001 - surface config/API issues in the UI
        report = f"_Report generation unavailable: {exc}_"

    return grade_md, result["processed_image"], confidences, report


with gr.Blocks(title="DR Clinical Report Generator") as demo:
    gr.Markdown("# 🩺 Diabetic Retinopathy - Clinical Report Generator")
    gr.Markdown(
        "Upload a retinal fundus image. A trained EfficientNet-B4 + Swin Transformer "
        "model grades diabetic-retinopathy severity (0-4), and Gemini drafts a "
        "structured clinical report around that grade."
    )
    gr.Markdown(DISCLAIMER)

    with gr.Row():
        with gr.Column():
            inp = gr.Image(type="pil", label="Upload fundus image")
            btn = gr.Button("Analyze", variant="primary")
        with gr.Column():
            grade_out = gr.Markdown()
            conf_out = gr.Label(num_top_classes=5, label="Severity probabilities")
            proc_out = gr.Image(label="Preprocessed (RFOV Crop + CLAHE)", type="pil")

    gr.Markdown("### Clinical report")
    report_out = gr.Markdown()

    btn.click(analyze, inputs=inp, outputs=[grade_out, proc_out, conf_out, report_out])


if __name__ == "__main__":
    demo.laun