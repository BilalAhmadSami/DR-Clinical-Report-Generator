"""Central configuration for the DR Clinical Report Generator.

Settings for the reused diabetic-retinopathy grading model, the training-matched
image preprocessing, and the Gemini report-generation model.
"""
from pathlib import Path
import os

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- DR grading model (reused from the trained DR Grading project) -------
# The 430 MB weights live in the public HuggingFace Space and are downloaded
# automatically on first run (no token needed).
DR_WEIGHTS_REPO = "DRG-Group-34/Diabetic_Retinopathy_Grading"
DR_WEIGHTS_REPO_TYPE = "space"
DR_WEIGHTS_FILE = "model_epoch_011_qwk_0.8121.pth"

EFFICIENTNET_NAME = "efficientnet_b4.ra2_in1k"
SWIN_NAME = "swin_base_patch4_window12_384.ms_in22k_ft_in1k"
NUM_CLASSES = 5
FUSION_HIDDEN_DIM = 1024
FUSION_DROPOUT = 0.3
IMAGE_SIZE = 384
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

CLASS_NAMES = {
    0: "No DR",
    1: "Mild DR",
    2: "Moderate DR",
    3: "Severe DR",
    4: "Proliferative DR",
}

# --- Preprocessing (matches training: crop -> pad -> resize -> CLAHE) ----
CROP_MODE = "auto"          # "auto" applies the fundus crop; "none" skips it
CROP_THRESHOLD = 15
CROP_MARGIN = 0.02
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = (8, 8)

# --- Gemini (vision-language LLM that writes the report) -----------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
