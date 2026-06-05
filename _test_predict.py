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
df_all = pd.read_csv("data/bogor_daily_val_safe.csv")
df_all['date'] = pd.to_datetime(df_all['date'])
df_all = df_all.sort_values('date').reset_index(drop=True)
df_all['time_idx'] = range(TRAIN_OFFSET, TRAIN_OFFSET + len(df_all))
df_all['group_id'] = 'Bogor'
for col in ['month', 'day_of_week', 'uvIndex']:
    if col in df_all.columns:
        df_all[col] = df_all[col].astype(str)
print(f"  Data rows: {len(df_all)} | Date range: {df_all['date'].min()} to {df_all['date'].max()}")

print("\n[4/6] Loading BMKG data...")
bmkg = pd.read_csv("data/juni.csv")
bmkg['TANGGAL'] = pd.to_datetime(bmkg['TANGGAL'], format='%d-%m-%Y')
bmkg = bmkg.sort_values('TANGGAL').reset_index(drop=True)
bmkg_7day = bmkg.head(PREDICTION_LENGTH).copy()
print(f"  BMKG rows: {len(bmkg)} | First 7 dates: {list(bmkg_7day['TANGGAL'].dt.date)}")

print("\n[5/6] Building forecast frame...")
forecast_dates = pd.date_range(BMKG_START_DATE, periods=PREDICTION_LENGTH, freq='D')
forecast_frame = df_all.copy()
existing_dates = set(forecast_frame['date'])
missing_dates = [d for d in forecast_dates if d not in existing_dates]
print(f"  Missing dates to add: {[str(d.date()) for d in missing_dates]}")

if missing_dates:
    last_row = forecast_frame.iloc[-1].copy()
    placeholder_rows = []
    for date in missing_dates:
        new_row = last_row.copy()
        new_row['date'] = date
        new_row['precipMM'] = 0.0
        new_row['month'] = str(date.month)
        new_row['day_of_week'] = str(date.dayofweek + 1)  # 1-indexed!
        new_row['day_of_year'] = date.dayofyear
        new_row['year'] = date.year
        placeholder_rows.append(new_row)
    forecast_frame = pd.concat([forecast_frame, pd.DataFrame(placeholder_rows)], ignore_index=True)

forecast_frame = forecast_frame.sort_values('date').reset_index(drop=True)
forecast_frame['time_idx'] = range(TRAIN_OFFSET, TRAIN_OFFSET + len(forecast_frame))
forecast_frame['group_id'] = 'Bogor'

for col in forecast_frame.columns:
    if col not in ['date', 'group_id'] and forecast_frame[col].dtype != 'object':
        forecast_frame[col] = forecast_frame[col].ffill().bfill()

for col in ['month', 'day_of_week', 'uvIndex']:
    if col in forecast_frame.columns:
        forecast_frame[col] = forecast_frame[col].astype(str)

print(f"  Forecast frame rows: {len(forecast_frame)}")
print(f"  Last 10 rows day_of_week: {list(forecast_frame['day_of_week'].tail(10))}")
print(f"  Last 10 rows group_id: {list(forecast_frame['group_id'].tail(10))}")

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
