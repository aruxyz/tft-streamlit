"""Quick local test: load model + metadata, build BMKG forecast, run prediction."""
import pickle
import io
import numpy as np
import pandas as pd
import torch
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet

ENCODER_LENGTH = 90
PREDICTION_LENGTH = 7
TRAIN_OFFSET = 4915
BMKG_START_DATE = pd.Timestamp('2025-06-01')

print("[1/6] Loading metadata...")
with open("models/dataset_metadata.pkl", "rb") as f:
    metadata = pickle.load(f)
print(f"  Metadata type: {type(metadata).__name__}")
print(f"  Reals: {len(metadata.reals)} | Categoricals: {len(metadata.categoricals)}")

# Check group encoder
cat_encoders = getattr(metadata, '_categorical_encoders', {})
for key, enc in cat_encoders.items():
    classes = getattr(enc, 'classes_', {})
    print(f"  Encoder '{key}': classes={classes}")

print("\n[2/6] Loading model checkpoint...")
checkpoint = torch.load(
    "models/tft_model_final_chronological.ckpt",
    map_location=torch.device("cpu"),
    weights_only=False
)
unknown_params = ['dataset_parameters', 'mask_bias', 'monotone_constraints']
for param in unknown_params:
    if param in checkpoint["hyper_parameters"]:
        del checkpoint["hyper_parameters"][param]

buffer = io.BytesIO()
torch.save(checkpoint, buffer)
buffer.seek(0)

model = TemporalFusionTransformer.load_from_checkpoint(buffer, map_location=torch.device("cpu"))
model.eval()
print("  Model loaded OK")

print("\n[3/6] Loading data...")
train_df = pd.read_csv("data/bogor_daily_train_safe.csv")
train_df['date'] = pd.to_datetime(train_df['date'])
train_df = train_df.sort_values('date').reset_index(drop=True)
train_df['time_idx'] = range(len(train_df))
train_df['group_id'] = 'Bogor'
for col in ['month', 'day_of_week', 'uvIndex']:
    if col in train_df.columns:
        train_df[col] = train_df[col].astype(str)

val_df = pd.read_csv("data/bogor_daily_val_safe.csv")
val_df['date'] = pd.to_datetime(val_df['date'])
val_df = val_df.sort_values('date').reset_index(drop=True)
val_df['time_idx'] = range(TRAIN_OFFSET, TRAIN_OFFSET + len(val_df))
val_df['group_id'] = 'Bogor'
for col in ['month', 'day_of_week', 'uvIndex']:
    if col in val_df.columns:
        val_df[col] = val_df[col].astype(str)

df_all = val_df.copy()
print(f"  Train rows: {len(train_df)} | Val rows: {len(val_df)}")

print("\n[4/6] Loading BMKG data...")
bmkg = pd.read_csv("data/juni.csv")
bmkg['TANGGAL'] = pd.to_datetime(bmkg['TANGGAL'], format='%d-%m-%Y')
bmkg = bmkg.sort_values('TANGGAL').reset_index(drop=True)
bmkg_7day = bmkg.head(PREDICTION_LENGTH).copy()
print(f"  BMKG rows: {len(bmkg)} | First 7 dates: {list(bmkg_7day['TANGGAL'].dt.date)}")

print("\n[5/6] Building forecast frame (train + val[-97:] like notebook)...")
# Persis seperti Kaggle notebook SS7
val_extended = val_df.copy()
forecast_dates = pd.date_range(BMKG_START_DATE, periods=PREDICTION_LENGTH, freq='D')
existing_dates = set(val_extended['date'])
missing_dates = [d for d in forecast_dates if d not in existing_dates]
print(f"  Missing dates to add: {[str(d.date()) for d in missing_dates]}")

if missing_dates:
    last_row = val_extended.iloc[-1].copy()
    placeholder_rows = []
    for date in missing_dates:
        new_row = last_row.copy()
        new_row['date'] = date
        new_row['precipMM'] = 0.0
        new_row['month'] = str(date.month)
        new_row['day_of_week'] = str(date.dayofweek + 1)
        new_row['day_of_year'] = date.dayofyear
        new_row['year'] = date.year
        placeholder_rows.append(new_row)
    val_extended = pd.concat([val_extended, pd.DataFrame(placeholder_rows)], ignore_index=True)

val_extended = val_extended.sort_values('date').reset_index(drop=True)
for col in val_extended.columns:
    if col not in ['date', 'group_id'] and val_extended[col].dtype != 'object':
        val_extended[col] = val_extended[col].ffill().bfill()
for col in ['month', 'day_of_week', 'uvIndex']:
    if col in val_extended.columns:
        val_extended[col] = val_extended[col].astype(str)

# concat train + last 97 baris val — PERSIS NOTEBOOK
forecast_frame = pd.concat(
    [train_df, val_extended.iloc[-(ENCODER_LENGTH + PREDICTION_LENGTH):]],
    ignore_index=True
)
for col in ['month', 'day_of_week', 'uvIndex', 'group_id']:
    if col in forecast_frame.columns:
        forecast_frame[col] = forecast_frame[col].astype(str)

print(f"  full_extended rows: {len(forecast_frame)}")
print(f"  Last 10 rows day_of_week: {list(forecast_frame['day_of_week'].tail(10))}")

print("\n[6/6] Running prediction...")
try:
    dataset = TimeSeriesDataSet.from_dataset(
        metadata, forecast_frame, predict=True, stop_randomization=True
    )
    print(f"  Dataset created: {len(dataset)} samples")

    dataloader = dataset.to_dataloader(train=False, batch_size=1, num_workers=0)
    result = model.predict(dataloader, mode="prediction", return_x=True)

    if hasattr(result, "output"):
        predictions_raw = result.output
    elif isinstance(result, tuple):
        predictions_raw = result[0]
    else:
        predictions_raw = result

    pred_arr = predictions_raw.detach().cpu().numpy()
    if len(pred_arr.shape) == 2:
        pred_arr = pred_arr[-1]
    elif len(pred_arr.shape) == 1:
        pred_arr = pred_arr[-PREDICTION_LENGTH:]
    pred_arr = pred_arr[:PREDICTION_LENGTH]
    predictions = np.maximum(pred_arr, 0)

    actual_7day = bmkg_7day['RR'].to_numpy()
    mae = float(np.mean(np.abs(predictions - actual_7day)))
    rmse = float(np.sqrt(np.mean((predictions - actual_7day) ** 2)))

    print("\n=== HASIL VALIDASI BMKG ===")
    print(f"{'Hari':5} {'Tanggal':12} {'Aktual':>8} {'Prediksi':>10} {'Error':>8}")
    print("-" * 50)
    for i in range(PREDICTION_LENGTH):
        tgl = bmkg_7day.iloc[i]['TANGGAL'].strftime('%d-%m-%Y')
        err = abs(predictions[i] - actual_7day[i])
        print(f"{i+1:5d} {tgl:12} {actual_7day[i]:8.2f} {predictions[i]:10.2f} {err:8.2f}")
    print("-" * 50)
    print(f"MAE  = {mae:.2f} mm/hari")
    print(f"RMSE = {rmse:.2f} mm/hari")
    print("\nSUCCESFULLY DONE!")
except Exception as e:
    print(f"\nERROR: {e}")
    import traceback
    traceback.print_exc()
