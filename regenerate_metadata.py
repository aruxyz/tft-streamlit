"""
Regenerate dataset_metadata.pkl for pytorch-forecasting >= 1.x compatibility.

The old metadata.pkl was pickled with pytorch-forecasting 0.10.3 + pandas 1.5.3.
It fails to load on pandas 2.x due to removed pandas.core.indexes.numeric module.
This script extracts dataset_parameters from the TFT checkpoint and creates a
fresh TimeSeriesDataSet using pytorch-forecasting >= 1.x with pandas 2.x.
"""

import pickle
import sys
import types
import pandas as pd
import torch
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer, NaNLabelEncoder


def _patch_pandas_compat():
    numeric_mod = types.ModuleType("pandas.core.indexes.numeric")
    numeric_mod.Int64Index = pd.Index
    numeric_mod.UInt64Index = pd.Index
    numeric_mod.Float64Index = pd.Index
    sys.modules["pandas.core.indexes.numeric"] = numeric_mod
    if not hasattr(pd.core.indexes, "numeric"):
        pd.core.indexes.numeric = numeric_mod  # type: ignore[attr-defined]


print("Patching pandas 1.x->2.x pickle compatibility...")
_patch_pandas_compat()

CHECKPOINT_PATH = "models/tft-leaky_chrono_h1-epoch=07-val_loss=2.937.ckpt"
OUTPUT_PATH = "models/dataset_metadata.pkl"
TRAIN_DATA_PATH = "data/bogor_daily_train_leaky_chrono.csv"

print(f"Loading checkpoint: {CHECKPOINT_PATH}")
checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
dp = checkpoint["dataset_parameters"]
print(f"dataset_parameters keys: {list(dp.keys())}")

print(f"Loading training data: {TRAIN_DATA_PATH}")
train_df = pd.read_csv(TRAIN_DATA_PATH)
train_df["date"] = pd.to_datetime(train_df["date"])
train_df = train_df.sort_values("date").reset_index(drop=True)
train_df["time_idx"] = range(len(train_df))
train_df["group_id"] = "Bogor"

for col in ["month", "day_of_week", "uvIndex", "is_extreme_rain"]:
    if col in train_df.columns:
        train_df[col] = train_df[col].astype(str)

protected_cols = ["encoder_length", "precipMM_center", "precipMM_scale", "relative_time_idx"]
for c in protected_cols:
    if c in train_df.columns:
        train_df = train_df.drop(columns=[c])

print(f"Training data: {len(train_df)} rows, {len(train_df.columns)} columns")

all_cats = (
    dp.get("time_varying_known_categoricals", [])
    + dp.get("time_varying_unknown_categoricals", [])
    + dp.get("static_categoricals", [])
)
fresh_cat_encoders = {cat: NaNLabelEncoder(add_nan=True) for cat in all_cats}
fresh_cat_encoders["group_id"] = NaNLabelEncoder(add_nan=True)
fresh_cat_encoders["__group_id__group_id"] = NaNLabelEncoder(add_nan=True)

auto_added_reals = {"encoder_length", "precipMM_center", "precipMM_scale", "relative_time_idx"}
filtered_static_reals = [r for r in dp.get("static_reals", []) if r not in auto_added_reals]
filtered_known_reals = [r for r in dp.get("time_varying_known_reals", []) if r not in auto_added_reals]

print("Creating TimeSeriesDataSet with fresh encoders...")
metadata = TimeSeriesDataSet(
    data=train_df,
    time_idx=dp["time_idx"],
    target=dp["target"],
    group_ids=dp["group_ids"],
    weight=dp.get("weight"),
    max_encoder_length=dp["max_encoder_length"],
    min_encoder_length=dp.get("min_encoder_length", dp["max_encoder_length"]),
    min_prediction_idx=dp.get("min_prediction_idx", 0),
    min_prediction_length=dp.get("min_prediction_length", 1),
    max_prediction_length=dp["max_prediction_length"],
    static_categoricals=dp.get("static_categoricals", []),
    static_reals=filtered_static_reals,
    time_varying_known_categoricals=dp.get("time_varying_known_categoricals", []),
    time_varying_known_reals=filtered_known_reals,
    time_varying_unknown_categoricals=dp.get("time_varying_unknown_categoricals", []),
    time_varying_unknown_reals=dp.get("time_varying_unknown_reals", []),
    variable_groups=dp.get("variable_groups", {}),
    constant_fill_strategy=dp.get("constant_fill_strategy", {}),
    allow_missing_timesteps=dp.get("allow_missing_timesteps", True),
    lags=dp.get("lags", {}),
    add_relative_time_idx=dp.get("add_relative_time_idx", True),
    add_target_scales=dp.get("add_target_scales", True),
    add_encoder_length=dp.get("add_encoder_length", True),
    target_normalizer=GroupNormalizer(
        method="standard",
        groups=dp["group_ids"],
        center=True,
        scale_by_group=False,
        transformation="log1p",
    ),
    categorical_encoders=fresh_cat_encoders,
    scalers={},
    randomize_length=None,
    predict_mode=False,
)

print(f"TimeSeriesDataSet: encoder={metadata.max_encoder_length}, "
      f"prediction={metadata.max_prediction_length}, target={metadata.target}")

print("Verifying from_dataset()...")
test_ds = TimeSeriesDataSet.from_dataset(metadata, train_df.tail(200), predict=True)
test_dl = test_ds.to_dataloader(train=False, batch_size=1, num_workers=0)
print(f"  OK - {len(test_dl)} batches")

print(f"Saving to: {OUTPUT_PATH}")
with open(OUTPUT_PATH, "wb") as f:
    pickle.dump(metadata, f)

print(f"\n[OK] dataset_metadata.pkl regenerated!")
print(f"  pytorch-forecasting: {__import__('pytorch_forecasting').__version__}")
print(f"  pandas: {pd.__version__}")
print(f"  torch: {torch.__version__}")
