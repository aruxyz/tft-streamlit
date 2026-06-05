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
        df = pd.read_csv("data/bogor_daily_val_safe.csv")
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        df['time_idx'] = range(len(df))
        df['group'] = 'bogor'
        return df
    except Exception as e:
        st.error(f"Gagal memuat data: {str(e)}")
        return None

df_all = load_data()

if df_all is None:
    st.stop()

min_date = df_all['date'].min() + timedelta(days=ENCODER_LENGTH)
max_date_limit = pd.Timestamp('2025-06-01')
max_date = min(df_all['date'].max() - timedelta(days=PREDICTION_LENGTH), max_date_limit)

valid_dates = df_all[
    (df_all['date'] >= min_date) &
    (df_all['date'] <= max_date)
]['date'].dt.date.unique()

valid_dates = sorted(valid_dates)

if len(valid_dates) == 0:
    st.warning("Tidak ada tanggal yang tersedia untuk simulasi.")
    st.stop()

selected_date = st.sidebar.selectbox(
    "Pilih Tanggal Awal Prediksi (H+1):",
    options=valid_dates,
    format_func=lambda x: x.strftime("%d %B %Y"),
    index=len(valid_dates) // 2,
    help=f"Model akan menggunakan data {ENCODER_LENGTH} hari sebelum tanggal ini sebagai input encoder"
)

st.sidebar.info(f"""
📋 **Informasi Prediksi:**
- Tanggal Prediksi: **{selected_date.strftime('%d %B %Y')}**
- Input Encoder: **{ENCODER_LENGTH} hari** data historis
- Output Decoder: **{PREDICTION_LENGTH} hari** prediksi ke depan
- Fitur Input: **25 variabel** cuaca & derived
""")

prediction_start = pd.Timestamp(selected_date)
encoder_start = prediction_start - timedelta(days=ENCODER_LENGTH)

df_encoder = df_all[
    (df_all['date'] >= encoder_start) &
    (df_all['date'] < prediction_start)
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
            df_pred = df_all[
                (df_all['date'] >= encoder_start) &
                (df_all['date'] < prediction_start + timedelta(days=PREDICTION_LENGTH))
            ].copy()

            df_pred['time_idx'] = range(len(df_pred))
            df_pred['group'] = 'bogor'

            dataset = TimeSeriesDataSet(
                df_pred,
                time_idx="time_idx",
                target="precipMM",
                group_ids=["group"],
                min_encoder_length=ENCODER_LENGTH,
                max_encoder_length=ENCODER_LENGTH,
                min_prediction_length=PREDICTION_LENGTH,
                max_prediction_length=PREDICTION_LENGTH,
                static_categoricals=["group"],
                time_varying_known_reals=["time_idx"],
                time_varying_unknown_reals=[
                    "precipMM", "maxtempC", "mintempC", "avgtempC",
                    "humidity", "pressure", "windspeedKmph",
                    "mean_precip_static", "std_precip_static",
                    "precip_lag_1", "precip_lag_3", "precip_lag_7",
                    "precip_rolling_3", "precip_rolling_7", "precip_rolling_30",
                    "precipMM_boxcox", "temp_range",
                    "humidity_temp_interaction", "pressure_change"
                ],
                add_relative_time_idx=True,
                add_target_scales=True,
                add_encoder_length=True,
                predict_mode=True,
            )

            val_dataloader = dataset.to_dataloader(train=False, batch_size=1, num_workers=0)

            progress_bar.progress(50)

            predictions_raw = model.predict(val_dataloader, mode="prediction")
            predictions_arr = predictions_raw.detach().cpu().numpy().flatten()

            progress_bar.progress(70)

            if len(predictions_arr) > PREDICTION_LENGTH:
                predictions_arr = predictions_arr[-PREDICTION_LENGTH:]

            predictions = np.maximum(predictions_arr, 0)
            p10 = np.maximum(predictions_arr * 0.8, 0)
            p50 = predictions
            p90 = np.maximum(predictions_arr * 1.2, 0)

            forecast_dates = [prediction_start + timedelta(days=i) for i in range(PREDICTION_LENGTH)]

            df_forecast = pd.DataFrame({
                'date': forecast_dates,
                'pred_p10': p10,
                'pred_p50': p50,
                'pred_p90': p90,
                'rainfall_mm': predictions,
                'type': 'prediksi'
            })

            df_historical = df_encoder[['date', 'precipMM']].copy()
            df_historical.columns = ['date', 'rainfall_mm']
            df_historical['type'] = 'historis'

            progress_bar.progress(100)
            st.success("Prediksi Berhasil dengan Model TFT!")

            col_chart, col_stats = st.columns([2, 1])

            with col_chart:
                st.subheader("Grafik Prediksi Curah Hujan")
                fig, ax = plt.subplots(figsize=(12, 6))

                ax.plot(
                    range(len(df_historical)),
                    df_historical['rainfall_mm'],
                    color='black',
                    linewidth=2,
                    label=f'Data Historis ({ENCODER_LENGTH} Hari)',
                    marker='o',
                    markersize=3
                )

                x_forecast = range(len(df_historical) - 1, len(df_historical) + PREDICTION_LENGTH - 1)

                ax.fill_between(
                    x_forecast, p10[:PREDICTION_LENGTH], p90[:PREDICTION_LENGTH],
                    alpha=0.2, color='steelblue', label='P10-P90 (Uncertainty Band)'
                )

                ax.plot(
                    x_forecast,
                    predictions,
                    color='red',
                    linewidth=2,
                    linestyle='--',
                    label=f'Prediksi TFT ({PREDICTION_LENGTH} Hari)',
                    marker='x',
                    markersize=6
                )

                ax.axvline(
                    x=len(df_historical) - 1,
                    color='gray',
                    linestyle=':',
                    linewidth=1,
                    alpha=0.5,
                    label='Sekarang'
                )

                ax.set_xlabel('Hari')
                ax.set_ylabel('Curah Hujan (mm)')
                ax.set_title('Forecast Curah Hujan Harian - Bogor (TFT)')
                ax.legend(loc='best')
                ax.grid(True, alpha=0.3)
                ax.set_xlim(-1, len(df_historical) + PREDICTION_LENGTH - 2)

                all_dates = list(df_historical['date'].dt.strftime('%d-%b')) + \
                           [d.strftime('%d-%b') for d in forecast_dates]
                ax.set_xticks(range(len(all_dates)))
                ax.set_xticklabels(all_dates, rotation=45, ha='right')

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

                tomorrow_rain = predictions[0]
                rain_cat, rain_emoji = bmkg_category(tomorrow_rain)

                st.metric(
                    f"{rain_emoji} Prediksi Besok (H+1)",
                    f"{tomorrow_rain:.1f} mm",
                    rain_cat,
                    delta_color="normal"
                )

                st.markdown("---")

                avg_rain_7d = np.mean(predictions)
                max_rain_day = np.argmax(predictions) + 1
                max_rain_val = max(predictions)

                col_m1, col_m2 = st.columns(2)

                col_m1.metric(
                    "Rata-rata 7 Hari",
                    f"{avg_rain_7d:.1f} mm"
                )

                col_m2.metric(
                    "Hari Terbasah",
                    f"H+{max_rain_day}",
                    f"{max_rain_val:.1f} mm"
                )

                st.markdown("### Tabel Prediksi Detail")
                df_table = pd.DataFrame({
                    'Hari': [f"H+{i+1}" for i in range(PREDICTION_LENGTH)],
                    'Tanggal': [d.strftime('%d %B %Y') for d in forecast_dates],
                    'P10 (min)': [f"{v:.1f}" for v in p10],
                    'P50 (median)': [f"{v:.1f}" for v in p50],
                    'P90 (max)': [f"{v:.1f}" for v in p90],
                    'Kategori': [bmkg_category(v)[0] for v in predictions],
                    'Emoji': [bmkg_category(v)[1] for v in predictions],
                })

                st.dataframe(
                    df_table.style.background_gradient(
                        cmap='Reds',
                        subset=['P50 (median)']
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
