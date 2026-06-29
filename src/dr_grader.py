"""DR grader: loads the trained EfficientNet-B4 + Swin (CORN) model and grades
a fundus image into one of five diabetic-retinopathy severity classes.

The model architecture and weights are reused from the original DR Grading
project; the weights are downloaded from the public HuggingFace Space on first
use. Inference uses the training-matched preprocessing (crop + CLAHE).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import timm
from torchvision import transforms
from huggingface_hub import hf_hub_download

from src.config import (
    DR_WEIGHTS_REPO, DR_WEIGHTS_REPO_TYPE, DR_WEIGHTS_FILE,
    EFFICIENTNET_NAME, SWIN_NAME, NUM_CLASSES, FUSION_HIDDEN_DIM, FUSION_DROPOUT,
    IMAGE_SIZE, NORM_MEAN, NORM_STD, CLASS_NAMES,
)
from src.preprocessing import preprocess_fundus


class ParallelEfficientNetSwinCORN(nn.Module):
    """EfficientNet-B4 + Swin Transformer parallel backbones with a CORN ordinal head.

    Must match the trained architecture exactly (layer names included) so the
    saved weights load with strict=True.
    """

    def __init__(self, efficientnet_name, swin_name, num_classes,
                 fusion_hidden_dim=1024, fusion_dropout=0.3, pretrained=False):
        super().__init__()
        self.efficientnet = timm.create_model(efficientnet_name, pretrained=pretrained, num_classes=0)
        self.swin = timm.create_model(swin_name, pretrained=pretrained, num_classes=0)
        eff_dim = getattr(self.efficientnet, "num_features", None)
        swin_dim = getattr(self.swin, "num_features", None)
        if eff_dim is None or swin_dim is None:
            raise ValueError("Could not determine backbone feature dimensions.")
        fusion_dim = eff_dim + swin_dim
        self.fusion = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(fusion_dropout),
            nn.Linear(fusion_hidden_dim, num_classes - 1),   # CORN: K-1 outputs
        )

    @staticmethod
    def _to_vector(feat):
        if feat.ndim == 2:
            return feat
        if feat.ndim == 3:
            return feat.mean(dim=1)
        if feat.ndim == 4:
            if feat.shape[1] > feat.shape[-1] and feat.shape[1] > feat.shape[-2]:
                return feat.mean(dim=(2, 3))
            return feat.mean(dim=(1, 2))
        raise ValueError(f"Unexpected feature shape: {feat.shape}")

    def forward(self, x):
        eff = self._to_vector(self.efficientnet(x))
        swin = self._to_vector(self.swin(x))
        return self.fusion(torch.cat([eff, swin], dim=1))


def _clean_state_dict(state_dict):
    """Strip any 'module.' prefixes left by DataParallel during training."""
    return {(k[len("module."):] if k.startswith("module.") else k): v
            for k, v in state_dict.items()}


_model = None
_device = None
_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(NORM_MEAN, NORM_STD),
])


def load_grader():
    """Load the model + weights once (cached for repeated calls)."""
    global _model, _device
    if _model is not None:
        return _model, _device
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Downloading DR weights from {DR_WEIGHTS_REPO} (first run only) ...")
    weights_path = hf_hub_download(
        repo_id=DR_WEIGHTS_REPO, filename=DR_WEIGHTS_FILE, repo_type=DR_WEIGHTS_REPO_TYPE,
    )
    model = ParallelEfficientNetSwinCORN(
        efficientnet_name=EFFICIENTNET_NAME, swin_name=SWIN_NAME,
        num_classes=NUM_CLASSES, fusion_hidden_dim=FUSION_HIDDEN_DIM,
        fusion_dropout=FUSION_DROPOUT, pretrained=False,
    )
    state_dict = _clean_state_dict(torch.load(weights_path, map_location=_device))
    model.load_state_dict(state_dict, strict=True)
    model.to(_device).eval()
    _model = model
    return _model, _device


def grade_fundus(pil_image):
    """Grade one fundus image.

    Returns the grade (0-4), its label, and the cumulative ordinal
    probabilities P(y > k). CORN decoding: the K-1 logits are conditional
    probabilities, so the cumulative P(y > k) is the cumulative product of
    their sigmoids, and the predicted grade is the count of cumulative
    probabilities above 0.5 (equivalent to coral_pytorch.corn_label_from_logits).
    """
    model, device = load_grader()
    processed = preprocess_fundus(pil_image)                 # training-matched crop + CLAHE
    x = _transform(processed).unsqueeze(0).to(device)
    with torch.inference_mode():
        logits = model(x)
        # CORN decoding: the K-1 outputs are CONDITIONAL probabilities
        # P(y>k | y>k-1), so the cumulative P(y>k) is the cumulative PRODUCT of
        # the sigmoids. The grade is the number of cumulative probs above 0.5.
        # This matches coral_pytorch.corn_label_from_logits exactly.
        probs = torch.cumprod(torch.sigmoid(logits), dim=1)  # P(grade > 0..3)
        grade = int((probs > 0.5).sum(dim=1).item())
    return {
        "grade": grade,
        "label": CLASS_NAMES[grade],
        "ordinal_probabilities": [round(p, 4) for p in probs.squeeze(0).cpu().tolist()],
        "processed_image": processed,
    }


if __name__ == "__main__":
    # Usage:  python -m src.dr_grader <path-to-fundus-image>
    import sys
    from PIL import Image

    if len(sys.argv) < 2:
        print("Usage: python -m src.dr_grader <path-to-fundus-image>")
        raise SystemExit(1)

    result = grade_fundus(Image.open(sys.argv[1]))
    print(f"\nGrade {result['grade']} - {result['label']}")
    print(f"Ordinal probabilities P(>0..>3): {result['ordinal_probabilities']}")
