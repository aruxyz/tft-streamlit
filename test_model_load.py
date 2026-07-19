"""
Test load model untuk memverifikasi fix mask_bias issue.
"""
import torch
import io
from pytorch_forecasting import TemporalFusionTransformer

print("Loading checkpoint...")
checkpoint = torch.load(
    "models/tft_model_final_chronological.ckpt",
    map_location=torch.device("cpu"),
    weights_only=False,
)

print(f"Original hyper_parameters keys: {list(checkpoint['hyper_parameters'].keys())}")

unknown_params = ['dataset_parameters', 'mask_bias', 'monotone_constraints']
for param in unknown_params:
    if param in checkpoint["hyper_parameters"]:
        print(f"Removing {param} from checkpoint...")
        del checkpoint["hyper_parameters"][param]

print(f"Filtered hyper_parameters keys: {list(checkpoint['hyper_parameters'].keys())}")

print("\nSaving filtered checkpoint to buffer...")
buffer = io.BytesIO()
torch.save(checkpoint, buffer)
buffer.seek(0)

print("Loading model from buffer...")
try:
    model = TemporalFusionTransformer.load_from_checkpoint(
        buffer,
        map_location=torch.device("cpu")
    )
    model.eval()
    print("SUCCESS: Model loaded!")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
