# -*- coding: utf-8 -*-
"""
IEMOCAP-specific hyper-params & paths
"""
import torch
from pathlib import Path

# ─── Data ────────────────────────────────────────────────────────────────
DATA_PATH   = Path("/home/user/MER/IEMOCAP/pkl_file/iemocap_multi_features.pkl")
NUM_CLASSES = 6
EMOTIONS    = ["happy", "sad", "neutral", "angry", "excited", "frustrated"]

FEATURE_DIMS = dict(text=1024, audio=1582, visual=342)
TEXT_FEATURE_KEY = "videoText1"

# ─── Model ───────────────────────────────────────────────────────────────
HGT_HIDDEN_CHANNELS = 768
HGT_NUM_HEADS       = 8
HGT_NUM_LAYERS      = 2
DROPOUT_RATE        = 0.10

# ─── Optim / training ────────────────────────────────────────────────────
LEARNING_RATE = 5e-6
BATCH_SIZE     = 16
EPOCHS         = 150
WEIGHT_DECAY   = 1e-5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WANDB_PROJECT="ICLR_IEMOCAP"

# ─── Graph metadata (identical for both datasets) ────────────────────────
NODE_TYPES = ["audio", "text", "visual"]
EDGE_TYPES = [
    # Speaker-aware temporal edges (unidirectional, past context only)
    ("text", "past_same", "text"),
    ("text", "past_other", "text"),
    ("audio", "past_same", "audio"),
    ("audio", "past_other", "audio"),
    ("visual", "past_same", "visual"),
    ("visual", "past_other", "visual"),
    
    # Cross-modal edges (remain the same)
    ("text","text_to_audio","audio"),  
    ("audio","audio_to_text","text"),
    ("text","text_to_visual","visual"),
    ("visual","visual_to_text","text"),
    ("audio","audio_to_visual","visual"),
    ("visual","visual_to_audio","audio"),
]
