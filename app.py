import streamlit as st
import pandas as pd
import torch
import pickle
import io
import numpy as np
import matplotlib.pyplot as plt
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
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🌧️ Bogor Rainfall Forecasting (TFT Model)")
st.write("Aplikasi demo skripsi untuk memprediksi curah hujan harian di Bogor menggunakan Temporal Fusion Transformer.")
st.markdown("---")

ENCODER_LENGTH = 90
PREDICTION_LENGTH = 7
TRAIN_OFFSET = 4915
BMKG_START_DATE = pd.Timestamp('2025-06-01')

@st.cache_resource
def load_model_and_metadata():
    try:
        with open("models/dataset_metadata.pkl", "rb") as f:
            metadata = pickle.load(f)

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

        model = TemporalFusionTransformer.load_from_checkpoint(
            buffer,
            map_location=torch.device("cpu")
        )
        model.eval()

        return model, metadata
    except Exception as e:
        st.error(f"Gagal memuat model: {str(e)}")
        raise

with st.spinner('Sedang memuat model AI...'):
    model, metadata = load_model_and_metadata()
    st.success("Model TFT & Metadata Berhasil Dimuat!")
    st.caption(f"Encoder: {ENCODER_LENGTH} hari | Prediksi: {PREDICTION_LENGTH} hari | Parameter: 393.559")

st.markdown("---")

@st.cache_data
def load_data():
    try:
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

        return train_df, val_df
    except Exception as e:
        st.error(f"Gagal memuat data: {str(e)}")
        return None, None

train_df, val_df = load_data()

if train_df is None or val_df is None:
    st.stop()

@st.cache_data
def load_bmkg_data():
    try:
        bmkg = pd.read_csv("data/juni.csv")
        bmkg['TANGGAL'] = pd.to_datetime(bmkg['TANGGAL'], format='%d-%m-%Y')
        bmkg = bmkg.sort_values('TANGGAL').reset_index(drop=True)
        return bmkg
    except Exception as e:
        st.error(f"Gagal memuat data aktual BMKG: {str(e)}")
        return None

bmkg_data = load_bmkg_data()

if bmkg_data is None:
    st.stop()

def build_bmkg_forecast_frame(val_df, train_df):
    val_extended = val_df.copy()
    forecast_dates = pd.date_range(BMKG_START_DATE, periods=PREDICTION_LENGTH, freq='D')
    existing_dates = set(val_extended['date'])
    missing_dates = [d for d in forecast_dates if d not in existing_dates]

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
        val_extended = pd.concat(
            [val_extended, pd.DataFrame(placeholder_rows)],
            ignore_index=True
        )

    val_extended = val_extended.sort_values('date').reset_index(drop=True)
    for col in val_extended.columns:
        if col not in ['date', 'group_id'] and val_extended[col].dtype != 'object':
            val_extended[col] = val_extended[col].ffill().bfill()
    for col in ['month', 'day_of_week', 'uvIndex']:
        if col in val_extended.columns:
            val_extended[col] = val_extended[col].astype(str)

    # Persis seperti Kaggle notebook SS7
    full_extended = pd.concat(
        [train_df, val_extended.iloc[-(ENCODER_LENGTH + PREDICTION_LENGTH):]],
        ignore_index=True
    )
    for col in ['month', 'day_of_week', 'uvIndex', 'group_id']:
        if col in full_extended.columns:
            full_extended[col] = full_extended[col].astype(str)

    return full_extended

selected_date = BMKG_START_DATE.date()

st.sidebar.info(f"""
📋 **Mode Validasi BMKG:**
- Periode Prediksi: **1-7 Juni 2025**
- Input Encoder: **{ENCODER_LENGTH} hari** data historis
- Output Decoder: **{PREDICTION_LENGTH} hari** prediksi ke depan
- Fitur Input: **25 variabel** cuaca & derived
- Ground Truth: **BMKG Stasiun Bogor (`juni.csv`)**
""")

prediction_start = pd.Timestamp(selected_date)
encoder_start = prediction_start - timedelta(days=ENCODER_LENGTH)

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
    st.subheader(f"Hasil Forecast: {PREDICTION_LENGTH} Hari Mulai {selected_date.strftime('%d %B %Y')}")

    with st.spinner('Sedang memproses prediksi dengan model TFT...'):
        progress_bar = st.progress(0)
        progress_bar.progress(20)

        try:
            df_pred = build_bmkg_forecast_frame(val_df, train_df)

            dataset = TimeSeriesDataSet.from_dataset(
                metadata,
                df_pred,
                predict=True,
                stop_randomization=True,
            )

            val_dataloader = dataset.to_dataloader(train=False, batch_size=1, num_workers=0)

            progress_bar.progress(50)

            prediction_result = model.predict(val_dataloader, mode="prediction", return_x=True)

            if hasattr(prediction_result, "output"):
                predictions_raw = prediction_result.output
            elif isinstance(prediction_result, tuple):
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
            p10 = np.maximum(predictions_arr * 0.8, 0)
            p50 = predictions
            p90 = np.maximum(predictions_arr * 1.2, 0)

            bmkg_7day = bmkg_data.head(PREDICTION_LENGTH).copy()
            forecast_dates = list(bmkg_7day['TANGGAL'])
            actual_7day = bmkg_7day['RR'].to_numpy()
            errors = np.abs(predictions - actual_7day)
            mae_bmkg = float(np.mean(errors))
            rmse_bmkg = float(np.sqrt(np.mean((predictions - actual_7day) ** 2)))

            df_forecast = pd.DataFrame({
                'date': forecast_dates,
                'pred_p10': p10,
                'pred_p50': p50,
                'pred_p90': p90,
                'actual_bmkg': actual_7day,
                'error_abs': errors,
                'rainfall_mm': predictions,
                'type': 'prediksi'
            })

            df_historical = df_encoder[['date', 'precipMM']].copy()
            df_historical.columns = ['date', 'rainfall_mm']
            df_historical['type'] = 'historis'

            progress_bar.progress(100)
            st.success(f"Prediksi Berhasil! MAE={mae_bmkg:.2f} mm/hari | RMSE={rmse_bmkg:.2f} mm/hari")

            col_chart, col_stats = st.columns([2, 1])

            with col_chart:
                st.subheader("Grafik Prediksi TFT vs Aktual BMKG")
                fig, ax = plt.subplots(figsize=(12, 6))

                days = list(range(1, PREDICTION_LENGTH + 1))

                ax.fill_between(
                    days,
                    np.minimum(actual_7day, predictions),
                    np.maximum(actual_7day, predictions),
                    alpha=0.2,
                    color='gray',
                    label='Area Error'
                )

                ax.plot(
                    days,
                    actual_7day,
                    color='black',
                    linewidth=2,
                    label='Aktual BMKG',
                    marker='o',
                    markersize=8
                )

                ax.plot(
                    days,
                    predictions,
                    color='steelblue',
                    linewidth=2,
                    label='Prediksi TFT',
                    marker='s',
                    markersize=8
                )

                ax.set_xlabel('Tanggal')
                ax.set_ylabel('Curah Hujan (mm)')
                ax.set_title(f'Prediksi TFT vs Aktual BMKG (1-7 Juni 2025)\nMAE={mae_bmkg:.2f} mm, RMSE={rmse_bmkg:.2f} mm')
                ax.legend(loc='best')
                ax.grid(True, alpha=0.3)
                ax.set_xlim(0.5, PREDICTION_LENGTH + 0.5)
                ax.set_xticks(days)
                ax.set_xticklabels([d.strftime('%d %b') for d in forecast_dates], rotation=45, ha='right')

                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

            with col_stats:
                st.subheader("Ringkasan Prediksi")

                def bmkg_category(mm):
                    if mm < 0.5:
                        return "Tidak Hujan", "☀️"
                    elif mm <= 20:
                        return "Hujan Ringan", "🌤️"
                    elif mm <= 50:
                        return "Hujan Sedang", "🌧️"
                    else:
                        return "Hujan Lebat", "⛈️"

                first_day_rain = predictions[0]
                rain_cat, rain_emoji = bmkg_category(first_day_rain)

                st.metric(
                    f"{rain_emoji} Prediksi 1 Juni",
                    f"{first_day_rain:.1f} mm",
                    rain_cat,
                    delta_color="normal"
                )

                st.markdown("---")

                avg_rain_7d = np.mean(predictions)
                max_rain_day = np.argmax(predictions) + 1
                max_rain_val = max(predictions)

                col_m1, col_m2 = st.columns(2)

                col_m1.metric(
                    "MAE vs BMKG",
                    f"{mae_bmkg:.2f} mm"
                )

                col_m2.metric(
                    "RMSE vs BMKG",
                    f"{rmse_bmkg:.2f} mm"
                )

                st.caption(f"Rata-rata prediksi: {avg_rain_7d:.1f} mm | Hari prediksi terbasah: H+{max_rain_day} ({max_rain_val:.1f} mm)")

                st.markdown("### Tabel Prediksi Detail")
                df_table = pd.DataFrame({
                    'Hari': [f"H+{i+1}" for i in range(PREDICTION_LENGTH)],
                    'Tanggal': [d.strftime('%d %B %Y') for d in forecast_dates],
                    'Aktual BMKG': np.round(actual_7day, 2),
                    'Prediksi TFT': np.round(predictions, 2),
                    'Error Absolut': np.round(errors, 2),
                    'Kategori': [bmkg_category(v)[0] for v in predictions],
                    'Emoji': [bmkg_category(v)[1] for v in predictions],
                })

                st.dataframe(
                    df_table.style.background_gradient(
                        cmap='Reds',
                        subset=['Error Absolut']
                    ),
                    use_container_width=True,
                    hide_index=True
                )

            st.markdown("---")
            st.subheader("Rekomendasi Berdasarkan Prediksi")
            rainy_days = sum(1 for p in predictions if p > 0.5)
            heavy_rain_days = sum(1 for p in predictions if p > 20)

            if rainy_days == 0:
                st.info("Prediksi cuaca cerah untuk 7 hari ke depan. Bagus untuk aktivitas luar ruangan!")
            elif rainy_days <= 2:
                st.warning("Diperkirakan beberapa hari berpotensi hujan ringan. Siapkan payung jika beraktivitas di luar.")
            elif rainy_days <= 4:
                st.warning("Prediksi cuaca cukup basah. Pertimbangkan untuk membawa jas hujan dan perlengkapan anti-air.")
            else:
                st.error("Prediksi curah hujan tinggi untuk minggu ini. Hindari aktivitas di luar dan waspada terhadap potensi banjir.")

            if heavy_rain_days > 0:
                st.error(f"Peringatan: Diperkirakan ada {heavy_rain_days} hari dengan hujan sedang/lebat. Harap berhati-hati!")

        except Exception as e:
            st.error(f"Terjadi kesalahan saat prediksi: {str(e)}")
            st.exception(e)

st.markdown("---")
st.caption("""
💻 **Powered by:** Temporal Fusion Transformer (TFT) + PyTorch Lightning + Streamlit
📊 **Dataset:** Data curah hujan harian Bogor (2008-2025, 6.122 observasi)
🎓 **Project:** Skripsi - Prediksi Curah Hujan dengan Deep Learning
""")
