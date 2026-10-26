"""M5: checkpoint-compatible DeepWide training and inference."""
from .config import DeepWideConfig, FoldArtifacts
from .model import build_model, load_model_checkpoint, save_model_checkpoint
from .features import open_embedding_store, iter_feature_batches
from .inference import predict_pairs
from .training import train_prepared_fold
from .checkpoint import checkpoint_sha256, inspect_checkpoint
