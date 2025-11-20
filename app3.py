# app.py (Versión con TEMA OSCURO + KPIs superiores, sin cambiar la lógica de modelos)
import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error

# -------------------------
# Página
# -------------------------
st.set_page_config(
    page_title="Forecast — GBM / Merton / Heston",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -------------------------
# Parâmetros fijos (ocultos)
# -------------------------
TWO_YEARS = 2
N_SIM = 10000        # fijo y no editable por UI
TEST_DAYS = 252      # fijo y no editable por UI
DT = 1 / 252

# -------------------------
# Tema oscuro (CSS)
# -------------------------
st.markdown(
    """
    <style>
    /* Fondo general y texto */
    .stApp, .css-18ni7ap { background: #0e1117; color: #e6eef3; }

    /* Tarjetas/KPIs */
    .card {
        background: #121419;
        border-radius: 10px;
        padding: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.6);
        border: 1px solid rgba(255,255,255,0.03);
    }
    .kpi-title { color: #9aa6b2; font-size:12px; margin-bottom:4px; }
    .kpi-value { color: #ffffff; font-size:20px; font-weight:700; }
    .kpi-sub { color: #8b98a6; font-size:12px; }

    /* Encabezados */
    h1 { color: #dbeafe; }
    .section-title { color: #dbeafe; font-weight:700; font-size:18px; }

    /* DataFrame oscuro (poco invasivo) */
    .stDataFrame table { background: #0f1519; color: #dbeafe; }
    .stTable table { background: #0f1519; color: #dbeafe; }

    /* Matplotlib canvas background */
    .stGraph div[data-testid="stCanvas"] canvas { background: #0e1117 !important; }

    /* Ajustes pequeños para textos */
    .css-1lcbmhc p, .css-1lcbmhc span { color: #cbd5e1; }

    /* Sidebar */
    .css-1d391kg { background-color: #0e1117; }

    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# Header (logo + title)
# -------------------------
left, right = st.columns([1, 8])
with left:
    # imagen incluida en la conversacion /mnt/data path (se usa como logo local)
    logo_path = "./forecast-analytics.png"
    try:
        st.image(logo_path, width=80)
    except Exception:
        # Si no existe la imagen local, no hacemos nada
        pass

with right:
    st.markdown("<h1 style='margin:0; padding:0;'>Forecast</h1>", unsafe_allow_html=True)
    st.markdown("<div style='color:#9aa6b2; margin-top:4px;'>Modelos: Browniano (GBM) · Merton · Heston — Rolling backtest</div>", unsafe_allow_html=True)

st.markdown("---")

# -------------------------
# Sidebar (información básica, oculto por defecto pero accesible)
# -------------------------
with st.sidebar:
    st.markdown("## Configuración de la app")
    st.write("Parámetros fijos para la demostración del producto.")
    st.markdown(f"- Simulaciones (N_SIM): **{N_SIM:,}**")
    st.markdown(f"- Window backtest (TEST_DAYS): **{TEST_DAYS} días**")
    st.markdown("---")
    st.markdown("**Fuente:** Yahoo Finance (datos diarios)")
    st.caption("La app usa parámetros fijos para consistencia en resultados (producción).")

# -------------------------
# Input principal (ticker)
# -------------------------
ticker = st.text_input("Ticker (Yahoo Finance)", value="AAPL").upper()

# -------------------------
# Descargar datos (2 años)
# -------------------------
@st.cache_data(ttl=3600)
def download_2y(sym):
    end = datetime.today().date()
    start = end - timedelta(days=365 * TWO_YEARS)
    df = yf.download(sym, start=start, end=end, progress=False)
    return df

df_raw = download_2y(ticker)
if df_raw is None or df_raw.empty:
    st.error("No se pudieron descargar datos. Verifica el ticker.")
    st.stop()

# seleccionar columna de precio
price_col = "Adj Close" if "Adj Close" in df_raw.columns else "Close"
if price_col not in df_raw.columns:
    st.error("No hay 'Close' ni 'Adj Close' en los datos.")
    st.stop()

data = df_raw[[price_col]].rename(columns={price_col: "price"}).dropna()
data["returns"] = np.log(data["price"] / data["price"].shift(1))
data.dropna(inplace=True)

# obtener nombre de empresa
try:
    info = yf.Ticker(ticker).info
    company_name = info.get("longName", info.get("shortName", ticker))
except Exception:
    company_name = ticker

# mostrar header compact con nombre de empresa y recuento
st.markdown(f"### <span style='color:#dbeafe'>{company_name} ({ticker})</span>", unsafe_allow_html=True)
st.markdown(f"<div style='color:#9aa6b2'>Datos históricos — {len(data)} registros (últimos {TWO_YEARS} años)</div>", unsafe_allow_html=True)

# -------------------------
# Validación datos mínimos
# -------------------------
if len(data) <= TEST_DAYS:
    st.error("No hay suficientes datos históricos para backtesting de 1 año.")
    st.stop()

train = data.iloc[:-TEST_DAYS].copy()
test = data.iloc[-TEST_DAYS:].copy()

# -------------------------
# Calibraciones (sin cambios)
# -------------------------
def calib_gbm_params(history_returns):
    mu_d = history_returns.mean()
    sigma_d = history_returns.std(ddof=1)
    mu = mu_d * 252
    sigma = sigma_d * np.sqrt(252)
    return float(mu), float(sigma)

def calib_merton_params(history_returns):
    r = history_returns.values
    r_ann = r * np.sqrt(252)
    sigma_r = np.std(r_ann, ddof=1)
    if sigma_r == 0:
        sigma_r = 1e-8
    thresh = 3 * sigma_r
    jumps_idx = np.where(np.abs(r) > thresh)[0]
    lam = (len(jumps_idx) / len(r)) * 252 if len(r) > 0 else 0.05
    if len(jumps_idx) > 0:
        mu_j = np.mean(r[jumps_idx])
        sigma_j = np.std(r[jumps_idx], ddof=1) if len(jumps_idx) > 1 else abs(r[jumps_idx].mean())*0.5
    else:
        mu_j = 0.0
        sigma_j = 0.02
        lam = 0.05
    return float(max(lam, 1e-6)), float(mu_j), float(max(sigma_j, 1e-6))

def calib_heston_params(history_returns):
    window = min(60, len(history_returns))
    rv = (history_returns * np.sqrt(252)).rolling(window=window).var().dropna()
    if len(rv) < 5:
        theta = history_returns.var()
        v0 = theta
        xi = 0.1
        rho = -0.3
    else:
        theta = float(rv.mean())
        v0 = float(rv.iloc[-1])
        xi = float(rv.std(ddof=1)) if rv.std(ddof=1) > 1e-6 else 0.1
        dv = rv.diff().dropna()
        minlen = min(len(history_returns.iloc[-len(dv):]), len(dv))
        try:
            rho = float(np.corrcoef(history_returns.dropna().values[-minlen:], dv.values[-minlen:])[0,1])
            if np.isnan(rho):
                rho = -0.3
        except Exception:
            rho = -0.3
    kappa = 1.5
    return float(v0), float(kappa), float(theta), float(max(xi, 1e-6)), float(rho)

# -------------------------
# Simuladores (sin cambios lógicos, usando N_SIM constante)
# -------------------------
def sim_gbm_1d(S_t, mu, sigma, n_sim=N_SIM):
    S_t = float(S_t)
    Z = np.random.normal(size=n_sim)
    log_ret = (mu - 0.5 * sigma**2) * DT + sigma * np.sqrt(DT) * Z
    return S_t * np.exp(np.asarray(log_ret))

def sim_merton_1d(S_t, mu, sigma, lam, mu_j, sigma_j, n_sim=N_SIM):
    S_t = float(S_t)
    Z = np.random.normal(size=n_sim)
    J = np.random.poisson(lam * DT, size=n_sim)
    jump = np.exp(mu_j + sigma_j * np.random.normal(size=n_sim)) - 1.0
    log_ret = (mu - 0.5 * sigma**2) * DT + sigma * np.sqrt(DT) * Z + J * jump
    return S_t * np.exp(np.asarray(log_ret))

def sim_heston_1d(S_t, mu, v_t, kappa, theta, xi, rho, n_sim=N_SIM):
    S_t = float(S_t)
    Z1 = np.random.normal(size=n_sim)
    Z2 = rho * Z1 + np.sqrt(max(0.0, 1 - rho**2)) * np.random.normal(size=n_sim)
    v_next = np.maximum(v_t + kappa * (theta - v_t) * DT + xi * np.sqrt(max(v_t,1e-12) * DT) * Z2, 1e-8)
    log_ret = (mu - 0.5 * v_t) * DT + np.sqrt(max(v_t,1e-12) * DT) * Z1
    S_next = S_t * np.exp(np.asarray(log_ret))
    return S_next, v_next

# -------------------------
# Rolling 1-day forecasting (no look-ahead)
# -------------------------
@st.cache_data
def run_rolling_forecast(full_data, test_index):
    preds = {"GBM": np.zeros(len(test_index)), "Merton": np.zeros(len(test_index)), "Heston": np.zeros(len(test_index))}
    dates = np.array(full_data.index[test_index])
    for idx_pos, idx in enumerate(test_index):
        history = full_data.iloc[:idx]
        if len(history) < 20:
            history = full_data.iloc[:idx]
        returns_hist = history["returns"].dropna()
        S_t = float(history["price"].iloc[-1])
        mu_r, sigma_r = calib_gbm_params(returns_hist)
        lam, mu_j, sigma_j = calib_merton_params(returns_hist)
        v0, kappa, theta, xi, rho = calib_heston_params(returns_hist)
        sim_gbm = sim_gbm_1d(S_t, mu_r, sigma_r)
        sim_mer = sim_merton_1d(S_t, mu_r, sigma_r, lam, mu_j, sigma_j)
        sim_hest, vnext = sim_heston_1d(S_t, mu_r, v0, kappa, theta, xi, rho)
        preds["GBM"][idx_pos] = float(np.mean(sim_gbm))
        preds["Merton"][idx_pos] = float(np.mean(sim_mer))
        preds["Heston"][idx_pos] = float(np.mean(sim_hest))
    return dates, preds

full_len = len(data)
start_test_pos = full_len - TEST_DAYS
test_positions = list(range(start_test_pos, full_len))

with st.spinner("Ejecutando rolling forecasts (esto puede tardar según N_SIM)..."):
    dates_arr, predictions = run_rolling_forecast(data, test_positions)

# -------------------------
# Alinear y calcular RMSE
# -------------------------
test_dates = pd.to_datetime(dates_arr)
test_prices = data.loc[test_dates, "price"].values

rmse_gbm = np.sqrt(mean_squared_error(test_prices, predictions["GBM"]))
rmse_merton = np.sqrt(mean_squared_error(test_prices, predictions["Merton"]))
rmse_heston = np.sqrt(mean_squared_error(test_prices, predictions["Heston"]))

results_df = pd.DataFrame({
    "Modelo": ["Browniano (GBM)", "Merton", "Heston"],
    "RMSE": [rmse_gbm, rmse_merton, rmse_heston]
}).sort_values("RMSE").reset_index(drop=True)

best_model = results_df.loc[0, "Modelo"]

# -------------------------
# Predicción final (recalibrada con ALL data) y percentiles
# -------------------------
hist_returns = data["returns"]
mu_final, sigma_final = calib_gbm_params(hist_returns)
lam_f, mu_j_f, sigma_j_f = calib_merton_params(hist_returns)
v0_f, kappa_f, theta_f, xi_f, rho_f = calib_heston_params(hist_returns)

ticker_yf = yf.Ticker(ticker)
try:
    S_last = float(ticker_yf.fast_info["last_price"])
except Exception:
    S_last = float(data["price"].iloc[-1])

s_gbm = sim_gbm_1d(S_last, mu_final, sigma_final, n_sim=N_SIM)
s_mer = sim_merton_1d(S_last, mu_final, sigma_final, lam_f, mu_j_f, sigma_j_f, n_sim=N_SIM)
s_hest, _ = sim_heston_1d(S_last, mu_final, v0_f, kappa_f, theta_f, xi_f, rho_f, n_sim=N_SIM)

mean_gbm = float(np.mean(s_gbm))
mean_mer = float(np.mean(s_mer))
mean_hest = float(np.mean(s_hest))

pred_table = pd.DataFrame({
    "Modelo": ["Browniano (GBM)", "Merton", "Heston"],
    "Predicción_esperada": [mean_gbm, mean_mer, mean_hest]
})

recommended_pred = pred_table.loc[pred_table["Modelo"] == best_model, "Predicción_esperada"].values[0]

if best_model.startswith("Browniano"):
    sims = s_gbm
elif best_model.startswith("Merton"):
    sims = s_mer
else:
    sims = s_hest

pct = np.percentile(sims, [5,25,50,75,95])

# -------------------------
# TOP KPIs (parte superior) - estilo tarjetas
# -------------------------
kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns([1.5,1.5,1.5,1.5], gap="large")

with kpi_col1:
    st.markdown(
        f"<div class='card'>"
        f"<div class='kpi-title'>Último precio</div>"
        f"<div class='kpi-value'>${S_last:,.2f}</div>"
        f"<div class='kpi-sub'>Precio de mercado (último)</div>"
        f"</div>",
        unsafe_allow_html=True
    )


with kpi_col2:
    st.markdown(
        f"<div class='card'>"
        f"<div class='kpi-title'>Mejor modelo (RMSE)</div>"
        f"<div class='kpi-value'>{best_model}</div>"
        f"<div class='kpi-sub'>RMSE: {results_df.loc[0,'RMSE']:.4f}</div>"
        f"</div>",
        unsafe_allow_html=True
    )

with kpi_col3:
    st.markdown(
        f"<div class='card'>"
        f"<div class='kpi-title'>Predicción (mañana)</div>"
        f"<div class='kpi-value'>${recommended_pred:,.2f}</div>"
        f"<div class='kpi-sub'>Valor esperado (modelo recomendado)</div>"
        f"</div>",
        unsafe_allow_html=True
    )

with kpi_col4:
    st.markdown(
        f"<div class='card'>"
        f"<div class='kpi-title'>Rango (p5 - p95)</div>"
        f"<div class='kpi-value'>${pct[0]:,.2f} — ${pct[-1]:,.2f}</div>"
        f"<div class='kpi-sub'>Intervalo de confianza</div>"
        f"</div>",
        unsafe_allow_html=True
    )

st.markdown("---")

# -------------------------
# MAIN: Gráficos y tablas
# -------------------------
left_col, right_col = st.columns([3,2])

with left_col:
    st.markdown("<div class='section-title'>Serie histórica y comparación (Backtest)</div>", unsafe_allow_html=True)

    # === GRÁFICO PRINCIPAL (FONDO BLANCO) ===
    fig, ax = plt.subplots(figsize=(12,5))

    ax.plot(test_dates, test_prices, label="Real", linewidth=2)
    ax.plot(test_dates, predictions["GBM"], label="GBM", alpha=0.9)
    ax.plot(test_dates, predictions["Merton"], label="Merton", alpha=0.9)
    ax.plot(test_dates, predictions["Heston"], label="Heston", alpha=0.9)

    ax.set_title(f"Comparación de Modelos (Backtesting) — {company_name} ({ticker})")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Precio")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()

    st.pyplot(fig)

    st.markdown("### Distribución - percentiles (modelo recomendado)")
    pct_df = pd.DataFrame({"Percentil":["5%","25%","50%","75%","95%"], "Precio":pct})
    st.dataframe(pct_df.style.set_properties(**{
        'background-color': '#0f1418',
        'color':'#dbeafe'
    }).format({"Precio":"${:,.2f}"}), use_container_width=True)

with right_col:
    st.markdown("<div class='section-title'>Resultados y Predicción</div>", unsafe_allow_html=True)
    st.dataframe(results_df.style.set_properties(**{
        'background-color': '#0f1418',
        'color': '#dbeafe'
    }).format({"RMSE":"{:.4f}"}), use_container_width=True)

    st.markdown("### Predicción esperada por modelo")
    st.dataframe(pred_table.style.set_properties(**{
        'background-color': '#0f1418',
        'color':'#dbeafe'
    }).format({"Predicción_esperada":"${:,.2f}"}), use_container_width=True)


st.markdown("---")


