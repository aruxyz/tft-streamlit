"""Quick local test: load TFT-H1 model + metadata, build forecast frame, run 1-day prediction.

Standalone verification script — does NOT launch Streamlit. Run from inside
the tft-streamlit/ directory:

    python _test_predict.py
"""
import os
import sys
import pickle
import numpy as np
import pandas as pd
import torch
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet

# Reconfigure stdout to UTF-8 so emoji in print statements don't crash on
# Windows consoles using legacy codepages (e.g. cp1252). No-op on systems
# where stdout is already UTF-8 (Streamlit's browser output is unaffected
# either way — this only matters for this standalone console script).
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ENCODER_LENGTH = 90
PREDICTION_LENGTH = 1
TRAIN_OFFSET = 4915
FORECAST_START_DATE = pd.Timestamp('2025-06-02')  # day after val_df ends (2025-06-01)

# Static validation metrics from TFT-H1 training (for reference / sanity print).
VAL_MAE = 5.3735
VAL_RMSE = 8.0754
N_PARAMS = 1220796
N_FEATURES = 28

print("=" * 70)
print("TFT-H1 PREDICTION TEST (horizon=1, leaky_chrono, QuantileLoss)")
print("=" * 70)

print("\n[1/5] Loading metadata...")
with open("models/dataset_metadata.pkl", "rb") as f:
    metadata = pickle.load(f)
print(f"  Metadata type: {type(metadata).__name__}")
print(f"  max_encoder_length: {metadata.max_encoder_length}")
print(f"  max_prediction_length: {metadata.max_prediction_length}")
print(f"  target: {metadata.target} | group_ids: {metadata.group_ids}")
print(f"  Reals ({len(metadata.reals)}): {metadata.reals}")
print(f"  Categoricals ({len(metadata.categoricals)}): {metadata.categoricals}")
print(f"  time_varying_unknown_categoricals: {metadata.time_varying_unknown_categoricals}")

# Check group encoder
cat_encoders = getattr(metadata, '_categorical_encoders', {})
for key, enc in cat_encoders.items():
    classes = getattr(enc, 'classes_', {})
    print(f"  Encoder '{key}': classes={classes}")

print("\n[2/5] Loading TFT-H1 model checkpoint...")
model = TemporalFusionTransformer.load_from_checkpoint(
    "models/tft-leaky_chrono_h1-epoch=07-val_loss=2.937.ckpt",
    map_location=torch.device("cpu"),
)
model.eval()
print("  Model loaded OK")

# Detect QuantileLoss (TFT-H1) vs point loss.
loss_quantiles = getattr(model.loss, 'quantiles', None)
has_real_quantiles = loss_quantiles is not None
model_quantiles = list(loss_quantiles) if has_real_quantiles else [0.5]
print(f"  loss type: {type(model.loss).__name__} | has_real_quantiles: {has_real_quantiles}")
print(f"  Model quantiles: {model_quantiles}")

n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Trainable params (from model): {n_trainable:,} (expected: {N_PARAMS:,})")

print("\n[3/5] Loading leaky_chrono data...")
train_df = pd.read_csv("data/bogor_daily_train_leaky_chrono.csv")
train_df['date'] = pd.to_datetime(train_df['date'])
train_df = train_df.sort_values('date').reset_index(drop=True)
train_df['time_idx'] = range(len(train_df))
train_df['group_id'] = 'Bogor'
for col in ['month', 'day_of_week', 'uvIndex', 'is_extreme_rain']:
    if col in train_df.columns:
        train_df[col] = train_df[col].astype(str)

val_df = pd.read_csv("data/bogor_daily_val_leaky_chrono.csv")
val_df['date'] = pd.to_datetime(val_df['date'])
val_df = val_df.sort_values('date').reset_index(drop=True)
val_df['time_idx'] = range(TRAIN_OFFSET, TRAIN_OFFSET + len(val_df))
val_df['group_id'] = 'Bogor'
for col in ['month', 'day_of_week', 'uvIndex', 'is_extreme_rain']:
    if col in val_df.columns:
        val_df[col] = val_df[col].astype(str)

print(f"  Train rows: {len(train_df)} | Val rows: {len(val_df)} | Total: {len(train_df) + len(val_df)}")
print(f"  Train date range: {train_df['date'].min().date()} -> {train_df['date'].max().date()}")
print(f"  Val   date range: {val_df['date'].min().date()} -> {val_df['date'].max().date()}")
print(f"  is_extreme_rain dtype (train): {train_df['is_extreme_rain'].dtype} | unique: {train_df['is_extreme_rain'].unique()[:5]}")

print(f"\n[4/5] Building forecast frame for {FORECAST_START_DATE.date()} (H+1)...")
forecast_dates = pd.date_range(FORECAST_START_DATE, periods=PREDICTION_LENGTH, freq='D')
encoder_start = FORECAST_START_DATE - pd.Timedelta(days=ENCODER_LENGTH)

val_extended = val_df.copy()
existing_dates = set(val_extended['date'])
missing_dates = [d for d in forecast_dates if d not in existing_dates]
print(f"  Missing dates to add as placeholders: {[str(d.date()) for d in missing_dates]}")

if missing_dates:
    last_row = val_extended.iloc[-1].copy()
    last_time_idx = int(last_row['time_idx'])
    placeholder_rows = []
    for i, date in enumerate(missing_dates, start=1):
        new_row = last_row.copy()
        new_row['date'] = date
        new_row['precipMM'] = 0.0
        new_row['month'] = str(date.month)
        new_row['day_of_week'] = str(date.dayofweek + 1)
        new_row['day_of_year'] = date.dayofyear
        new_row['year'] = date.year
        new_row['time_idx'] = last_time_idx + i
        new_row['is_extreme_rain'] = '0'
        placeholder_rows.append(new_row)
    val_extended = pd.concat([val_extended, pd.DataFrame(placeholder_rows)], ignore_index=True)

val_extended = val_extended.sort_values('date').reset_index(drop=True)
for col in val_extended.columns:
    if col not in ['date', 'group_id'] and val_extended[col].dtype != 'object':
        val_extended[col] = val_extended[col].ffill().bfill()
for col in ['month', 'day_of_week', 'uvIndex', 'is_extreme_rain']:
    if col in val_extended.columns:
        val_extended[col] = val_extended[col].astype(str)

last_forecast_date = forecast_dates[-1]
val_window = val_extended[
    (val_extended['date'] >= encoder_start) &
    (val_extended['date'] <= last_forecast_date)
].copy()
forecast_frame = pd.concat([train_df, val_window], ignore_index=True)
for col in ['month', 'day_of_week', 'uvIndex', 'is_extreme_rain', 'group_id']:
    if col in forecast_frame.columns:
        forecast_frame[col] = forecast_frame[col].astype(str)

print(f"  forecast_frame rows: {len(forecast_frame)}")
print(f"  Last 3 rows [date, time_idx, precipMM, is_extreme_rain]:")
print(forecast_frame[['date', 'time_idx', 'precipMM', 'is_extreme_rain']].tail(3).to_string(index=False))

print("\n[5/5] Running prediction...")
try:
    dataset = TimeSeriesDataSet.from_dataset(
        metadata, forecast_frame, predict=True, stop_randomization=True
    )
    print(f"  Dataset created: {len(dataset)} samples")

    dataloader = dataset.to_dataloader(train=False, batch_size=1, num_workers=0)

    def _find_q(q_target, qs, tol=1e-6):
        for i, q in enumerate(qs):
            if abs(float(q) - q_target) < tol:
                return i
        return None

    band_source = "real_quantile" if has_real_quantiles else "empirical_fallback"

    if has_real_quantiles:
        print("  Using mode=quantiles (real QuantileLoss heads)")
        result = model.predict(dataloader, mode="quantiles", return_x=True)
        if hasattr(result, "output"):
            quantiles_raw = result.output
        elif isinstance(result, (tuple, list)):
            quantiles_raw = result[0]
        else:
            quantiles_raw = result
        q_arr = quantiles_raw.detach().cpu().numpy()
        print(f"  Raw quantile output shape: {q_arr.shape}")
        if len(q_arr.shape) == 3:
            q_arr = q_arr[-1]
        elif len(q_arr.shape) == 2:
            q_arr = q_arr[-PREDICTION_LENGTH:, :]
        q_arr = q_arr[:PREDICTION_LENGTH, :]
        p10_idx = _find_q(0.1, model_quantiles)
        p50_idx = _find_q(0.5, model_quantiles)
        p90_idx = _find_q(0.9, model_quantiles)
        if p10_idx is None: p10_idx = 1
        if p50_idx is None: p50_idx = 3
        if p90_idx is None: p90_idx = 5
        print(f"  Indices — P10={p10_idx}, P50={p50_idx}, P90={p90_idx}")
        p10 = np.maximum(q_arr[:, p10_idx], 0)
        p50 = np.maximum(q_arr[:, p50_idx], 0)
        p90 = np.maximum(q_arr[:, p90_idx], 0)
        predictions = p50
    else:
        print("  Fallback: mode=prediction (MAE-trained, no quantile heads)")
        result = model.predict(dataloader, mode="prediction", return_x=True)
        if hasattr(result, "output"):
            predictions_raw = result.output
        elif isinstance(result, (tuple, list)):
            predictions_raw = result[0]
        else:
            predictions_raw = result
        pred_arr = predictions_raw.detach().cpu().numpy()
        print(f"  Raw prediction output shape: {pred_arr.shape}")
        if len(pred_arr.shape) == 2:
            pred_arr = pred_arr[-1]
        elif len(pred_arr.shape) == 1:
            pred_arr = pred_arr[-PREDICTION_LENGTH:]
        pred_arr = pred_arr[:PREDICTION_LENGTH]
        predictions = np.maximum(pred_arr, 0)
        p50 = predictions
        encoder_precip = forecast_frame['precipMM'].to_numpy().astype(float)[-ENCODER_LENGTH:]
        sigma_enc = float(np.std(encoder_precip)) if len(encoder_precip) > 1 else 0.0
        band_half = 1.28 * sigma_enc
        p10 = np.maximum(predictions - band_half, 0)
        p90 = np.maximum(predictions + band_half, 0)
        print(f"  Empirical band half-width: {band_half:.2f} mm (sigma_encoder={sigma_enc:.2f})")

    # --- Single-day result summary (horizon=1) ---
    pred_val = float(predictions[0])
    p10_val = float(p10[0])
    p50_val = float(p50[0])
    p90_val = float(p90[0])

    def rain_category(mm):
        if mm < 0.5:
            return "Tidak Hujan", "☀️"
        elif mm <= 20:
            return "Hujan Ringan", "🌤️"
        elif mm <= 50:
            return "Hujan Sedang", "🌧️"
        else:
            return "Hujan Lebat", "⛈️"

    cat, emoji = rain_category(p50_val)

    print(f"\n{'=' * 70}")
    print(f"HASIL PREDIKSI TFT-H1 untuk {FORECAST_START_DATE.strftime('%d-%m-%Y')} (H+1)")
    print(f"{'=' * 70}")
    print(f"  Tanggal       : {FORECAST_START_DATE.strftime('%d %B %Y')}")
    print(f"  P10           : {p10_val:.2f} mm")
    print(f"  P50 (point)   : {p50_val:.2f} mm")
    print(f"  P90           : {p90_val:.2f} mm")
    print(f"  Band width    : {p90_val - p10_val:.2f} mm")
    print(f"  Kategori      : {emoji} {cat}")
    print(f"  Band source   : {band_source}")
    print(f"\n  Static validation metrics (from TFT-H1 training):")
    print(f"    MAE  = {VAL_MAE} mm/hari")
    print(f"    RMSE = {VAL_RMSE} mm/hari")
    print(f"    Trainable params = {N_PARAMS:,} (verified from model: {n_trainable:,})")
    print(f"    Features         = {N_FEATURES}")
    print(f"\n  SUCCESFULLY DONE!")
except Exception as e:
    print(f"\nERROR: {e}")
    import traceback
    traceback.print_exc()
