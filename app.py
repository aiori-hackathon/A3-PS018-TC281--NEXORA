import streamlit as st
import pandas as pd
import plotly.express as px
from src.channel_model import Scenario, simulate_series
from src.train_eval import train_and_evaluate, forecast_next

st.set_page_config(page_title="HAPS Link Intelligence | PS-018", page_icon="📡", layout="wide")

# Professional dashboard styling
st.markdown("""
<style>
    .stApp {
        background-color: #F5F7FB;
    }

    h1 {
        color: #142B4A;
        font-weight: 800;
    }

    h2, h3 {
        color: #203D63;
    }

    [data-testid="stMetric"] {
        background-color: white;
        border: 1px solid #DCE5F0;
        padding: 20px;
        border-radius: 14px;
        box-shadow: 0 3px 12px rgba(20, 43, 74, 0.06);
    }

    [data-testid="stSidebar"] {
        background-color: #EAF0F8;
    }

    div[data-testid="stAlert"] {
        border-radius: 12px;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

st.title("📡 HAPS Link-Quality Prediction")
st.caption("PS-018 · AIORI Hackathon 3 · 6G & Future Networks")
st.warning("Prototype notice: this demo uses a documented simplified channel model and synthetic scenarios. It is not yet a validated ITU-R implementation or a field-tested HAPS system.")

with st.sidebar:
    st.header("Scenario controls")
    distance_km = st.slider("Slant distance (km)", 20.0, 500.0, 80.0, 5.0)
    frequency_ghz = st.select_slider("Carrier frequency (GHz)", options=[2.0, 6.0, 12.0, 20.0, 28.0, 38.0], value=28.0)
    elevation_deg = st.slider("Elevation angle (degrees)", 5, 90, 45)
    rain_mm_h = st.slider("Rain rate (mm/h)", 0.0, 100.0, 10.0, 1.0)
    cloud_loss_db = st.slider("Cloud attenuation (dB)", 0.0, 10.0, 1.0, 0.5)
    gas_loss_db = st.slider("Gas attenuation (dB)", 0.0, 5.0, 0.5, 0.1)
    drift_m = st.slider("Platform lateral drift (m)", 0.0, 200.0, 15.0, 5.0)
    jitter_deg = st.slider("Pointing jitter (degrees)", 0.0, 3.0, 0.4, 0.1)
    horizon = st.slider("Forecast horizon (steps)", 1, 12, 5)

scenario = Scenario(
    distance_km=distance_km, frequency_ghz=frequency_ghz,
    elevation_deg=float(elevation_deg), rain_mm_h=rain_mm_h,
    cloud_loss_db=cloud_loss_db, gas_loss_db=gas_loss_db,
    drift_m=drift_m, jitter_deg=jitter_deg
)
series = simulate_series(scenario, steps=120, seed=42)
result = train_and_evaluate(seed=42)
forecast = forecast_next(series, result["model"], horizon=horizon)

latest_snr = float(series["snr_db"].iloc[-1])
threshold = 5.0
risk = "HIGH" if forecast["min_forecast_snr_db"] < threshold else ("MEDIUM" if forecast["min_forecast_snr_db"] < threshold + 3 else "LOW")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Current simulated SNR", f"{latest_snr:.2f} dB")
c2.metric(f"Forecast minimum ({horizon} steps)", f"{forecast['min_forecast_snr_db']:.2f} dB")
c3.metric("Estimated outage risk", risk)
c4.metric("Validation MAE", f"{result['mae_db']:.2f} dB", help="Synthetic held-out test set; not real-world validation")

plot_df = series[["step", "snr_db"]].copy()
plot_df["series"] = "Simulated SNR"
future_df = pd.DataFrame({"step": range(120, 120 + horizon), "snr_db": forecast["values"], "series": "ML forecast"})
chart_df = pd.concat([plot_df, future_df], ignore_index=True)
fig = px.line(
    chart_df,
    x="step",
    y="snr_db",
    color="series",
    title="Simulated Link Quality & AI Forecast",
    template="plotly_white",
    markers=False,
    color_discrete_map={
        "Simulated SNR": "#1769FF",
        "ML forecast": "#00B894"
    },
    labels={
        "step": "Time Step",
        "snr_db": "Signal-to-Noise Ratio (dB)",
        "series": "Series"
    }
)

fig.update_layout(
    title_font_size=20,
    title_font_color="#142B4A",
    font=dict(family="Arial", size=13, color="#334155"),
    paper_bgcolor="white",
    plot_bgcolor="white",
    margin=dict(l=20, r=20, t=70, b=20),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    hovermode="x unified"
)

fig.update_xaxes(showgrid=False)
fig.update_yaxes(gridcolor="#E5EAF2", zeroline=False)
fig.add_hline(y=threshold, line_dash="dash", annotation_text="Example outage threshold (5 dB)")
st.plotly_chart(fig, use_container_width=True)

st.subheader("📡 Live Signal Quality Indicator")

gauge_fig = px.pie(
    names=["Current SNR", "Remaining Scale"],
    values=[max(0, min(latest_snr, 40)), 40 - max(0, min(latest_snr, 40))],
    hole=0.75,
    color_discrete_sequence=["#1769FF", "#E5EAF2"]
)

gauge_fig.update_traces(
    textinfo="none",
    sort=False,
    direction="clockwise",
    showlegend=False
)

gauge_fig.add_annotation(
    text=f"<b>{latest_snr:.2f} dB</b><br><span style='font-size:13px'>Simulated SNR</span>",
    x=0.5,
    y=0.5,
    showarrow=False,
    font=dict(size=23, color="#142B4A")
)

gauge_fig.update_layout(
    showlegend=False,
    height=280,
    margin=dict(l=20, r=20, t=20, b=20),
    paper_bgcolor="white"
)

st.plotly_chart(gauge_fig, use_container_width=True)

st.caption(
    "The gauge uses a display scale of 0–40 dB. "
    "It is a visualization, not a validated signal-quality rating."
)


left, right = st.columns(2)
with left:
    st.subheader("Scenario summary")
    st.dataframe(pd.DataFrame([
        ("Slant distance", f"{distance_km:.1f} km"),
        ("Carrier frequency", f"{frequency_ghz:.1f} GHz"),
        ("Elevation", f"{elevation_deg}°"),
        ("Rain rate", f"{rain_mm_h:.1f} mm/h"),
        ("Cloud attenuation input", f"{cloud_loss_db:.1f} dB"),
        ("Gas attenuation input", f"{gas_loss_db:.1f} dB"),
        ("Platform drift", f"{drift_m:.1f} m"),
        ("Pointing jitter", f"{jitter_deg:.1f}°"),
    ], columns=["Parameter", "Value"]), hide_index=True, use_container_width=True)
with right:
    st.subheader("Suggested action")
    if risk == "HIGH":
        st.error("High simulated outage risk: consider a more robust modulation/coding profile, link adaptation, or handover evaluation.")
    elif risk == "MEDIUM":
        st.warning("Moderate simulated risk: monitor the link and prepare link adaptation or handover if the trend continues.")
    else:
        st.success("Low simulated risk for this scenario: continue monitoring and re-evaluate when conditions change.")
    st.write("Main modeled drivers:")
    st.write("- Frequency and distance affect free-space path loss.")
    st.write("- Rain, cloud and gas inputs add attenuation.")
    st.write("- Drift and jitter add simplified pointing loss.")
    st.write("- The ML model forecasts a synthetic series; SHAP explanations are a planned extension, not included in this starter.")

st.subheader("AI Link Intelligence")

insight_col1, insight_col2 = st.columns(2)

with insight_col1:
    st.markdown("### 📡 Link condition")
    if risk == "HIGH":
        st.error(
            "The forecast crosses the example outage threshold. "
            "Review link conditions and evaluate mitigation options."
        )
    elif risk == "MEDIUM":
        st.warning(
            "The forecast is approaching the example outage threshold. "
            "Continue monitoring the predicted SNR."
        )
    else:
        st.success(
            "The forecast remains above the example outage threshold. "
            "Continue monitoring as conditions change."
        )

with insight_col2:
    st.markdown("### 🔎 Forecast summary")
    st.write(f"Current simulated SNR: **{latest_snr:.2f} dB**")
    st.write(
        f"Minimum forecast SNR: "
        f"**{forecast['min_forecast_snr_db']:.2f} dB**"
    )
    st.write(f"Forecast horizon: **{horizon} steps**")
    st.caption(
        "These insights use synthetic data and a demo threshold, "
        "not validated real-world HAPS measurements."
    )


st.subheader("Model Performance Comparison")

metrics_df = result["metrics"].copy()
st.dataframe(metrics_df, hide_index=True, use_container_width=True)

error_columns = [
    col for col in metrics_df.columns
    if col.strip().upper() in ["MAE (DB)", "RMSE (DB)", "MAE", "RMSE"]
]

if error_columns:
    st.markdown("### Prediction Error Comparison")
    chart_metrics = metrics_df.melt(
        id_vars=metrics_df.columns[0],
        value_vars=error_columns,
        var_name="Metric",
        value_name="Error (dB)"
    )

    fig_metrics = px.bar(
        chart_metrics,
        x=metrics_df.columns[0],
        y="Error (dB)",
        color="Metric",
        barmode="group",
        title="Lower error indicates better predictions",
        template="plotly_white",
        color_discrete_sequence=["#1769FF", "#00B894"]
    )

    fig_metrics.update_layout(
        xaxis_title="Model",
        yaxis_title="Error (dB)",
        legend_title="Metric"
    )

    st.plotly_chart(fig_metrics, use_container_width=True)
else:
    st.info("Prediction-error chart unavailable: check the metric column names.")

st.download_button("Download generated scenario CSV", data=series.to_csv(index=False).encode("utf-8"),
                   file_name="haps_scenario_series.csv", mime="text/csv")
st.caption("All thresholds and channel terms are configurable demo assumptions. Replace the simplified attenuation terms with verified ITU-R calculations before making scientific performance claims.")

st.subheader("🌦️ Scenario Risk Analysis")

risk_scenarios = [
    {
        "Scenario": "Current conditions",
        "Rain (mm/h)": rain_mm_h,
        "Cloud loss (dB)": cloud_loss_db,
        "Gas loss (dB)": gas_loss_db
    },
    {
        "Scenario": "Heavier rain",
        "Rain (mm/h)": min(rain_mm_h + 20, 100),
        "Cloud loss (dB)": cloud_loss_db,
        "Gas loss (dB)": gas_loss_db
    },
    {
        "Scenario": "Higher cloud attenuation",
        "Rain (mm/h)": rain_mm_h,
        "Cloud loss (dB)": min(cloud_loss_db + 3, 10),
        "Gas loss (dB)": gas_loss_db
    }
]

risk_df = pd.DataFrame(risk_scenarios)
st.dataframe(risk_df, hide_index=True, use_container_width=True)

st.caption(
    "These are illustrative input scenarios only. "
    "Their actual SNR and outage effects require running the channel model "
    "for each scenario; the table does not claim to predict those effects."
)
