"""NOMAD-EEG: Native Omni-Montage Adaptive Diffusion.

    import nomad_eeg
    model = nomad_eeg.load_pretrained()                  # downloads the released weights once
    feats = nomad_eeg.extract_features(model, x, channels)   # x: [N, E, T] preprocessed, 200 Hz
"""
from .model import CHANNELS_21, SplitSSM, SplitSSMConfig, channel_coords
from .classifier import Classifier
from .hub import CORPUS_ID, WEIGHTS_URL, load_pretrained
from .features import extract_features
from .preprocessing import FS, RobustScaler, preprocess

__all__ = ["CHANNELS_21", "CORPUS_ID", "Classifier", "FS", "RobustScaler", "SplitSSM",
           "SplitSSMConfig", "WEIGHTS_URL", "channel_coords", "extract_features",
           "load_pretrained", "preprocess"]
__version__ = "1.0.0"
