"""
Regenerate metadata.pkl from model checkpoint to fix pickle compatibility issue.

Problem: Old metadata.pkl was created with different pytorch-forecasting version
Solution: Regenerate using dataset_parameters from checkpoint
"""

import pickle
import torch
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer

# Load checkpoint to get dataset parameters
print("Loading checkpoint...")
checkpoint = torch.load(
    'models/tft_model_final_chronological.ckpt',
    map_location='cpu',
    weights_only=False,
)

# Extract dataset parameters
dataset_params = checkpoint['hyper_parameters']['dataset_parameters']
print(f"Dataset parameters found: {list(dataset_params.keys())}")

# Create a minimal metadata dict with dataset parameters
metadata = {
    'dataset_parameters': dataset_params,
    'max_encoder_length': dataset_params['max_encoder_length'],
    'max_prediction_length': dataset_params['max_prediction_length'],
    'time_idx': dataset_params['time_idx'],
    'target': dataset_params['target'],
    'group_ids': dataset_params['group_ids'],
    'static_categoricals': dataset_params.get('static_categoricals', []),
    'static_reals': dataset_params.get('static_reals', []),
    'time_varying_known_categoricals': dataset_params.get('time_varying_known_categoricals', []),
    'time_varying_known_reals': dataset_params.get('time_varying_known_reals', []),
    'time_varying_unknown_categoricals': dataset_params.get('time_varying_unknown_categoricals', []),
    'time_varying_unknown_reals': dataset_params.get('time_varying_unknown_reals', []),
    'categorical_encoders': dataset_params.get('categorical_encoders', {}),
    'scalers': dataset_params.get('scalers', {}),
}

# Save new metadata.pkl
print("Saving new metadata.pkl...")
with open('models/dataset_metadata.pkl', 'wb') as f:
    pickle.dump(metadata, f)

print("[OK] metadata.pkl regenerated successfully!")
print(f"Metadata keys: {list(metadata.keys())}")
