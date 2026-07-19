import streamlit as st
import pandas as pd
import torch
import pickle
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

try:
    from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
    from pytorch_forecasting.data import GroupNormalizer
except ImportError:
    st.error("Library pytorch_forecasting belum terinstall. Jalankan: pip install -r requirements.txt")
    st.stop()

st.set_page_config(
    page_title="Bogor Rain Forecast",
    layout="wide",
)

st.title("Bogor Rainfall Forecasting")
st.write("Aplikasi demo skripsi untuk memprediksi curah hujan harian di Bogor menggunakan Temporal Fusion Transformer (horizon 1 hari).")
st.markdown("---")

ENCODER_LENGTH = 90
PREDICTION_LENGTH = 1
TRAIN_OFFSET = 4915
DEFAULT_START_DATE = pd.Timestamp('2025-06-01')

# Static validation metrics from TFT-H1 training (summary.json), for display
# alongside each forecast so users see the model's general error margin.
VAL_MAE = 5.3735
VAL_RMSE = 8.0754
VAL_N = 1236


def _fmt_id(n):
    """Format integer with dot thousands separator (Indonesian style)."""
    return f"{n:,}".replace(",", ".")


@st.cache_resource
def load_model_and_metadata():
    try:
        with open("models/dataset_metadata.pkl", "rb") as f:
            metadata = pickle.load(f)

        # TFT-H1 checkpoint: 1-day horizon, QuantileLoss, leaky_chrono split.
        # Loads cleanly under pytorch-forecasting==0.10.3.
        model = TemporalFusionTransformer.load_from_checkpoint(
            "models/tft-leaky_chrono_h1-epoch=07-val_loss=2.937.ckpt",
            map_location=torch.device("cpu"),
        )
        model.eval()

        # Detect whether the model was trained with QuantileLoss (can produce
        # real P10/P50/P90) or a point loss like MAE (only median available).
        # TFT-H1 uses QuantileLoss with quantiles [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98].
        loss_quantiles = getattr(model.loss, 'quantiles', None)
        if loss_quantiles is not None:
            has_real_quantiles = True
            model_quantiles = list(loss_quantiles)
        else:
            # Model trained with point loss (MAE/MSE/etc) — no quantile heads.
            # mode="quantiles" will return shape [..., 1] (only the median).
            has_real_quantiles = False
            model_quantiles = [0.5]

        return model, metadata, model_quantiles, has_real_quantiles
    except Exception as e:
        st.error(f"Gagal memuat model: {str(e)}")
        raise

with st.spinner('Sedang memuat model AI...'):
    model, metadata, model_quantiles, has_real_quantiles = load_model_and_metadata()
    st.success("Model & Metadata Berhasil Dimuat!")

st.markdown("---")

@st.cache_data
def load_data():
    try:
        train_df = pd.read_csv("data/bogor_daily_train_leaky_chrono.csv")
        train_df['date'] = pd.to_datetime(train_df['date'])
        train_df = train_df.sort_values('date').reset_index(drop=True)
        train_df['time_idx'] = range(len(train_df))
        train_df['group_id'] = 'Bogor'
        # Categoricals must be string dtype for TimeSeriesDataSet.
        # month/day_of_week/uvIndex = time_varying_known_categoricals,
        # is_extreme_rain = time_varying_unknown_categoricals (TFT-H1).
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

        return train_df, val_df
    except Exception as e:
        st.error(f"Gagal memuat data: {str(e)}")
        return None, None

train_df, val_df = load_data()

if train_df is None or val_df is None:
    st.stop()

def build_forecast_frame(val_df, train_df, start_date):
    """Build the extended dataframe used for TFT prediction.

    Slices the encoder window by date (`>= encoder_start`) and appends
    placeholder rows for forecast dates beyond val_df's last observed date.
    Placeholder rows increment `time_idx` from the last observed row so the
    `time_idx` column stays contiguous per group (required by
    TimeSeriesDataSet).
    """
    forecast_dates = pd.date_range(start_date, periods=PREDICTION_LENGTH, freq='D')
    encoder_start = start_date - pd.Timedelta(days=ENCODER_LENGTH)

    val_extended = val_df.copy()
    last_observed_date = val_extended['date'].max()
    last_forecast_date = forecast_dates[-1]

    # Fill EVERY day between the last observed date and the last forecast
    # date (inclusive), not just the forecast target date(s). If the user
    # picks a date more than 1 day beyond val_df's last observed date, only
    # placeholder-ing the target date(s) leaves a hole in the time series
    # (e.g. val ends 2025-06-01, target=2025-07-01 -> 2025-06-02..06-30
    # would be missing), which breaks TimeSeriesDataSet's contiguous
    # time_idx requirement: the encoder needs ENCODER_LENGTH unbroken days
    # immediately before the prediction window, or from_dataset(predict=True)
    # raises "filters should not remove all entries".
    if last_forecast_date > last_observed_date:
        fill_dates = pd.date_range(last_observed_date + pd.Timedelta(days=1), last_forecast_date, freq='D')
    else:
        fill_dates = pd.DatetimeIndex([])

    existing_dates = set(val_extended['date'])
    missing_dates = [d for d in fill_dates if d not in existing_dates]

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
            # is_extreme_rain is a time_varying_unknown_categorical in TFT-H1;
            # placeholder rows set it to '0' (string) before the dtype cast below.
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

    # Date-based slice: encoder_start to the last forecast date. Truncating
    # at forecast_dates[-1] ensures the user's selected forecast window is
    # always the LAST PREDICTION_LENGTH rows — which is what TFT's
    # from_dataset(predict=True) uses as the prediction target. Without this
    # upper bound, picking a historical date would cause TFT to predict for
    # val_df's tail instead of the selected date.
    last_forecast_date = forecast_dates[-1]
    val_window = val_extended[
        (val_extended['date'] >= encoder_start) &
        (val_extended['date'] <= last_forecast_date)
    ].copy()
    full_extended = pd.concat([train_df, val_window], ignore_index=True)
    for col in ['month', 'day_of_week', 'uvIndex', 'is_extreme_rain', 'group_id']:
        if col in full_extended.columns:
            full_extended[col] = full_extended[col].astype(str)

    return full_extended

# --- Sidebar: date picker ---------------------------------------------------
# min_value: need at least ENCODER_LENGTH days of history before the start date
# max_value: allow predicting up to 30 days beyond val_df's last observed date
min_date = (val_df['date'].min() + timedelta(days=ENCODER_LENGTH)).date()
max_date = (val_df['date'].max() + timedelta(days=30)).date()
default_date = DEFAULT_START_DATE.date()
if default_date < min_date:
    default_date = min_date
if default_date > max_date:
    default_date = max_date

st.markdown("### Pilih Tanggal Prediksi")
selected_date = st.date_input(
    "Tanggal Prediksi",
    value=default_date,
    min_value=min_date,
    max_value=max_date,
    help=(
        f"Prediksi 1 hari ke depan untuk tanggal ini. "
        f"Encoder menggunakan {ENCODER_LENGTH} hari historis sebelum tanggal ini. "
        f"Rentang valid: {min_date.strftime('%d %b %Y')} s/d {max_date.strftime('%d %b %Y')}."
    )
)

prediction_start = pd.Timestamp(selected_date)
encoder_start = prediction_start - timedelta(days=ENCODER_LENGTH)

total_obs = len(train_df) + len(val_df)

df_encoder = val_df[
    (val_df['date'] >= encoder_start) &
    (val_df['date'] < prediction_start)
].copy()

with st.expander(f"Lihat Data Input ({ENCODER_LENGTH} Hari Terakhir)", expanded=False):
    display_cols = ['date', 'precipMM', 'maxtempC', 'mintempC', 'humidity', 'pressure', 'windspeedKmph']
    st.dataframe(
        df_encoder[display_cols].style.background_gradient(cmap='Blues', subset=['precipMM']),
        use_container_width=True
    )

col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    predict_btn = st.button(
        "Jalankan Prediksi",
        use_container_width=True,
        type="primary"
    )

if predict_btn:
    st.markdown("---")
    st.subheader(f"Hasil Forecast untuk {prediction_start.strftime('%d %B %Y')}")

    with st.spinner('Sedang memproses prediksi...'):
        progress_bar = st.progress(0)
        progress_bar.progress(20)

        try:
            df_pred = build_forecast_frame(val_df, train_df, prediction_start)

            dataset = TimeSeriesDataSet.from_dataset(
                metadata,
                df_pred,
                predict=True,
                stop_randomization=True,
            )

            val_dataloader = dataset.to_dataloader(train=False, batch_size=1, num_workers=0)

            progress_bar.progress(50)

            # TFT-H1 was trained with QuantileLoss, so mode="quantiles" returns
            # shape [batch, horizon, n_quantiles] with n_quantiles = 7
            # ([0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]). Only P50 (median, idx 3)
            # is used as the point prediction; the other quantiles are not
            # surfaced in the UI.
            #
            # Fallback path: model trained with a point loss (MAE/MSE/etc),
            # mode="quantiles" would return shape [..., 1] (only the median),
            # so we use mode="prediction" directly instead.

            def _find_q_idx(q_target, qs, tol=1e-6):
                for i, q in enumerate(qs):
                    if abs(float(q) - q_target) < tol:
                        return i
                return None

            if has_real_quantiles:
                prediction_result = model.predict(val_dataloader, mode="quantiles", return_x=True)

                if hasattr(prediction_result, "output"):
                    quantiles_raw = prediction_result.output
                elif isinstance(prediction_result, (tuple, list)):
                    quantiles_raw = prediction_result[0]
                else:
                    quantiles_raw = prediction_result

                quantiles_arr = quantiles_raw.detach().cpu().numpy()

                # Expected shape: [batch, horizon, n_quantiles]. Take last sample.
                if len(quantiles_arr.shape) == 3:
                    quantiles_arr = quantiles_arr[-1]  # [horizon, n_quantiles]
                elif len(quantiles_arr.shape) == 2:
                    quantiles_arr = quantiles_arr[-PREDICTION_LENGTH:, :]

                quantiles_arr = quantiles_arr[:PREDICTION_LENGTH, :]

                p50_idx = _find_q_idx(0.5, model_quantiles)
                if p50_idx is None: p50_idx = 3

                progress_bar.progress(70)

                p50 = np.maximum(quantiles_arr[:, p50_idx], 0)
                predictions = p50  # P50 as point prediction
            else:
                # Fallback: point-loss model. Use mode="prediction" directly.
                prediction_result = model.predict(val_dataloader, mode="prediction", return_x=True)

                if hasattr(prediction_result, "output"):
                    predictions_raw = prediction_result.output
                elif isinstance(prediction_result, (tuple, list)):
                    predictions_raw = prediction_result[0]
                else:
                    predictions_raw = prediction_result

                predictions_arr = predictions_raw.detach().cpu().numpy()
                if len(predictions_arr.shape) == 2:
                    predictions_arr = predictions_arr[-1]
                elif len(predictions_arr.shape) == 1:
                    predictions_arr = predictions_arr[-PREDICTION_LENGTH:]
                predictions_arr = predictions_arr[:PREDICTION_LENGTH]

                progress_bar.progress(70)

                predictions = np.maximum(predictions_arr, 0)
                p50 = predictions

            progress_bar.progress(100)

            pred_val = float(predictions[0])
            p50_val = float(p50[0])

            st.success(
                f"Prediksi Berhasil untuk {prediction_start.strftime('%d %B %Y')}! "
                f"Prediksi = {p50_val:.2f} mm."
            )

            st.subheader("Ringkasan Prediksi")

            def rain_category(mm):
                """Klasifikasi intensitas curah hujan berdasarkan ambang resmi."""
                if mm < 0.5:
                    return "Tidak Hujan"
                elif mm <= 20:
                    return "Hujan Ringan"
                elif mm <= 50:
                    return "Hujan Sedang"
                else:
                    return "Hujan Lebat"

            rain_cat = rain_category(p50_val)

            col_main, col_mae, col_rmse = st.columns([2, 1, 1])
            with col_main:
                st.metric(
                    f"Prediksi {prediction_start.strftime('%d %b')}",
                    f"{p50_val:.1f} mm",
                    rain_cat,
                    delta_color="normal"
                )
            with col_mae:
                st.metric("MAE Model", f"{VAL_MAE:.2f} mm/hari")
            with col_rmse:
                st.metric("RMSE Model", f"{VAL_RMSE:.2f} mm/hari")

            st.markdown("---")
            st.markdown("### Tabel Prediksi Detail")
            df_table = pd.DataFrame({
                'Tanggal': [prediction_start.strftime('%d %B %Y')],
                'Prediksi (mm)': [round(p50_val, 2)],
                'Kategori': [rain_cat],
            })
            st.dataframe(
                df_table,
                use_container_width=True,
                hide_index=True
            )

            st.markdown("---")
            st.subheader("Rekomendasi Berdasarkan Prediksi")
            date_label = prediction_start.strftime('%d %B %Y')
            rain_today = p50_val > 0.5
            heavy_rain_today = p50_val > 20

            if not rain_today:
                st.info(f"Prediksi untuk {date_label}: cuaca cerah (Tidak Hujan, <0.5 mm). Bagus untuk aktivitas luar ruangan!")
            elif p50_val <= 20:
                st.warning(f"Prediksi untuk {date_label}: hujan ringan ({p50_val:.1f} mm). Siapkan payung jika beraktivitas di luar.")
            elif p50_val <= 50:
                st.warning(f"Prediksi untuk {date_label}: hujan sedang ({p50_val:.1f} mm). Pertimbangkan untuk membawa jas hujan dan perlengkapan anti-air.")
            else:
                st.error(f"Prediksi untuk {date_label}: curah hujan tinggi ({p50_val:.1f} mm). Hindari aktivitas di luar dan waspada terhadap potensi banjir.")

            if heavy_rain_today:
                st.error(f"Peringatan: Prediksi hujan sedang/lebat untuk {date_label} ({p50_val:.1f} mm). Harap berhati-hati!")

        except Exception as e:
            st.error(f"Terjadi kesalahan saat prediksi: {str(e)}")
            st.exception(e)

st.markdown("---")
st.caption(f"""
**Powered by:** Temporal Fusion Transformer + PyTorch Lightning + Streamlit
**Dataset:** Data curah hujan harian Bogor (2008-2025, {_fmt_id(total_obs)} observasi — train {_fmt_id(len(train_df))} + val {_fmt_id(len(val_df))})
**Project:** Skripsi - Prediksi Curah Hujan dengan Deep Learning
""")
