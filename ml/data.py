"""Data generation and persistence utilities."""

from . import core

# Re-export selected APIs from core

generate_negative_samples = core.generate_negative_samples
generate_balanced_elliott_dataset = core.generate_balanced_elliott_dataset
save_dataset = core.save_dataset
load_dataset = core.load_dataset
