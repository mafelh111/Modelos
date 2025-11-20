import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error
from scipy.stats import norm
from scipy.optimize import minimize

# -------------------------
# Configuración de Página
# -------------------------
st.set_page_config(
    page_title="Forecast Financedamus",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -------------------------
# Parámetros
# -------------------------
TWO_YEARS = 2
TEST_DAYS = 60     # Reducido ligeramente para que la demo no sea eterna con Heston QMLE
N_SIM_FINAL = 10000  # Precisión máxima para el dato final
N_SIM_BACKTEST = 500 # Precisión media para el gráfico (velocidad)
DT = 1 / 252

# -------------------------
# Estilos CSS (Dark Mode)
# -------------------------
st.markdown(
    """
    <style>
    .stApp, .css-18ni7ap { background: #0e1117; color: #e6eef3; }
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
    h1 { color: #dbeafe; }
    .section-title { color: #dbeafe; font-weight:700; font-size:18px; }
    .stDataFrame table { background: #0f1519; color: #dbeafe; }
    .stGraph div[data-testid="stCanvas"] canvas { background: #0e1117 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# HEADER
# -------------------------
left, right = st.columns([1, 8])
with left:
    try:
        st.image("./forecast-analytics.png", width=80)
    except:
        pass
with right:
    st.markdown("<h1 style='margin:0; padding:0;'>Financedamus</h1>", unsafe_allow_html=True)
    st.markdown("<div style='color:#9aa6b2; margin-top:4px;'>Modelos: Browniano (GBM) · Merton · Heston — Rolling backtest</div>", unsafe_allow_html=True)

st.markdown("---")

# ==============================================================================
#  BLOQUE 1: CLASES DE MODELADO ESTOCÁSTICO (CORREGIDAS Y OPTIMIZADAS)
# ==============================================================================

class GBM:
    """ Geometric Brownian Motion calibrado con MLE """
    def __init__(self):
        self.mu = None
        self.sigma = None

    def fit(self, prices):
        # r_t = ln(Pt / Pt-1)
        log_returns = np.log(prices[1:] / prices[:-1])
        self.mu = np.mean(log_returns) / DT
        self.sigma = np.std(log_returns, ddof=1) / np.sqrt(DT)

    def simulate(self, S0, dt=DT, N=10000):
        Z = np.random.normal(0, 1, N)
        # Drift y difusión exactos
        return S0 * np.exp((self.mu - 0.5 * self.sigma**2) * dt + self.sigma * np.sqrt(dt) * Z)

    def predict_expectation(self, S0, dt=DT, N=10000):
        return np.mean(self.simulate(S0, dt, N))

class Merton:
    """ Merton Jump Diffusion calibrado con MLE (Mezcla Gaussiana + Poisson) """
    def __init__(self, mu=0.0, sigma=0.2, lam=0.1, mu_J=0.0, sigma_J=0.1):
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.lam = float(lam)
        self.mu_J = float(mu_J)
        self.sigma_J = float(sigma_J)

    def _log_likelihood(self, params, log_returns, dt):
        mu, sigma, lam, mu_J, sigma_J = params
        if sigma <= 0 or lam < 0 or sigma_J <= 0: return np.inf
        
        mu_dt = (mu - lam * (np.exp(mu_J + 0.5*sigma_J**2) - 1)) * dt
        sigma_dt = sigma * np.sqrt(dt)
        
        p0 = np.exp(-lam * dt)
        p1 = lam * dt * p0
        
        # Aproximación de 2 saltos max para velocidad
        pdf = (p0 * norm.pdf(log_returns, loc=mu_dt, scale=sigma_dt) +
               p1 * norm.pdf(log_returns, loc=mu_dt + mu_J, scale=np.sqrt(sigma_dt**2 + sigma_J**2)))
        return -np.sum(np.log(np.maximum(pdf, 1e-10)))

    def fit(self, prices):
        log_returns = np.log(prices[1:] / prices[:-1])
        # Estimaciones iniciales heurísticas para ayudar al optimizador
        sigma_init = np.std(log_returns)/np.sqrt(DT)
        x0 = [0.1, sigma_init, 0.5, -0.02, 0.05] 
        bounds = [(-2, 2), (0.01, 2), (0, 10), (-1, 1), (0.001, 1)]
        
        try:
            res = minimize(self._log_likelihood, x0, args=(log_returns, DT), method="L-BFGS-B", bounds=bounds)
            self.mu, self.sigma, self.lam, self.mu_J, self.sigma_J = res.x
        except:
            # Fallback si falla la optimización
            self.mu, self.sigma, self.lam, self.mu_J, self.sigma_J = 0.1, sigma_init, 0.0, 0.0, 0.0

    def simulate(self, S0, dt=DT, N=10000):
        Z = np.random.normal(0, 1, N)
        J_num = np.random.poisson(self.lam * dt, N)
        J_val = np.random.normal(self.mu_J, self.sigma_J, N) * J_num # Simplificación vectorizada segura
        
        drift = (self.mu - 0.5*self.sigma**2 - self.lam*(np.exp(self.mu_J+0.5*self.sigma_J**2)-1)) * dt
        diffusion = self.sigma * np.sqrt(dt) * Z
        return S0 * np.exp(drift + diffusion + J_val)

    def predict_expectation(self, S0, dt=DT, N=10000):
        return np.mean(self.simulate(S0, dt, N))

class Heston:
    """ Heston Stochastic Volatility con QMLE y Euler-Maruyama Sub-stepping """
    def __init__(self):
        self.kappa, self.theta, self.xi, self.rho, self.v0, self.mu = None, None, None, None, None, None

    def _q_likelihood(self, params, log_returns, dt):
        mu, kappa, theta, xi, rho, v0 = params
        if xi<=0 or theta<=0 or v0<=0 or kappa<=0 or abs(rho)>=0.99: return np.inf
        
        n = len(log_returns)
        v = v0
        ll = 0.0
        # Loop optimizado simple
        for t in range(n):
            var_pred = max(v, 1e-8)
            ll += 0.5*np.log(var_pred) + 0.5*((log_returns[t] - mu*dt)**2)/(var_pred*dt)
            z_proxy = (log_returns[t] - mu*dt)/np.sqrt(var_pred*dt)
            v = v + kappa*(theta - v)*dt + xi*np.sqrt(var_pred*dt)*z_proxy # Aproximación determinista del path de varianza para QMLE
        return ll

    def fit(self, prices):
        log_returns = np.log(prices[1:] / prices[:-1])
        var_hist = np.var(log_returns)/DT
        
        # Initial guess
        x0 = [0.1, 2.0, var_hist, 0.3, -0.5, var_hist]
        bounds = [(-1,1), (0.1,10), (0.001,1), (0.01,2), (-0.99,0.99), (0.001,1)]
        
        try:
            # Usar solo últimos 500 días para calibrar Heston (por velocidad) si hay muchos datos
            data_to_fit = log_returns[-500:] if len(log_returns) > 500 else log_returns
            res = minimize(self._q_likelihood, x0, args=(data_to_fit, DT), method="L-BFGS-B", bounds=bounds)
            self.mu, self.kappa, self.theta, self.xi, self.rho, self.v0 = res.x
        except:
            self.mu, self.kappa, self.theta, self.xi, self.rho, self.v0 = 0.1, 1.0, var_hist, 0.1, -0.5, var_hist

    def simulate(self, S0, dt=DT, N=10000, substeps=5): # Substeps reducido a 20 para balance
        dt_sub = dt/substeps
        S = np.full(N, S0, dtype=float)
        v = np.full(N, self.v0, dtype=float)
        
        for _ in range(substeps):
            Z1 = np.random.normal(size=N)
            Z2 = self.rho*Z1 + np.sqrt(1-self.rho**2)*np.random.normal(size=N)
            
            v = np.maximum(v + self.kappa*(self.theta - v)*dt_sub + self.xi*np.sqrt(v)*np.sqrt(dt_sub)*Z2, 1e-8)
            S *= np.exp((self.mu - 0.5*v)*dt_sub + np.sqrt(v)*np.sqrt(dt_sub)*Z1)
        return S

    def predict_expectation(self, S0, dt=DT, N=10000):
        return np.mean(self.simulate(S0, dt, N))

# ==============================================================================
#  BLOQUE 2: LÓGICA DE APLICACIÓN
# ==============================================================================

ticker = st.text_input("Ticker (Yahoo Finance)", value="MSFT").upper()

@st.cache_data(ttl=3600)
def download_data(sym):
    end = datetime.today().date()
    start = end - timedelta(days=365 * TWO_YEARS)
    df = yf.download(sym, start=start, end=end, progress=False)
    return df

df_raw = download_data(ticker)

if df_raw is None or df_raw.empty or df_raw.shape[0] < 60:
    st.error("Error al descargar datos o historial insuficiente.")
    st.stop()

price_col = "Adj Close" if "Adj Close" in df_raw.columns else "Close"
data = df_raw[[price_col]].rename(columns={price_col: "price"}).dropna()

# Info empresa
try:
    info = yf.Ticker(ticker).info
    company_name = info.get("longName", ticker)
except:
    company_name = ticker

st.markdown(f"### <span style='color:#dbeafe'>{company_name} ({ticker})</span>", unsafe_allow_html=True)

# Validación de longitud para Backtest
if len(data) <= TEST_DAYS:
    st.warning(f"Datos insuficientes para {TEST_DAYS} días de prueba. Usando 30 días.")
    TEST_DAYS = 30

train = data.iloc[:-TEST_DAYS]
test = data.iloc[-TEST_DAYS:]

# ------------------------------------------------
#  Rolling Forecast con las Nuevas Clases (OOP)
# ------------------------------------------------
@st.cache_data(show_spinner=False)
def run_rolling_forecast_oop(full_data, test_days):
    preds = {"GBM": [], "Merton": [], "Heston": []}
    
    # Índices de prueba
    test_indices = range(len(full_data) - test_days, len(full_data))
    dates = full_data.index[test_indices]
    
    # Barra de progreso (Heston QMLE es lento)
    progress_bar = st.progress(0)
    total = len(test_indices)
    
    # Instancias reutilizables (opcional, pero limpio crear nuevas para asegurar stateless)
    for i, idx in enumerate(test_indices):
        # Ventana deslizante: tomamos todo hasta ayer
        history_prices = full_data["price"].iloc[max(0, idx-250):idx].values

        S_t = history_prices[-1]
        
        # --- GBM ---
        gbm = GBM()
        gbm.fit(history_prices)
        p_g = gbm.predict_expectation(S_t, N=N_SIM_BACKTEST)
        
        # --- Merton ---
        mer = Merton()
        if i == 0:
            mer = Merton()
            mer.fit(history_prices)
        else:
            # Refit solo cada 5 días
            if i % 5 == 0:
                mer.fit(history_prices)

        p_m = mer.predict_expectation(S_t, N=N_SIM_BACKTEST)
        
        # --- Heston ---
        # --- Heston (solo calibrar 1 vez) ---
        if i == 0:
            hes = Heston()
            hes.fit(full_data["price"].values[:idx])

        p_h = hes.predict_expectation(S_t, N=N_SIM_BACKTEST)

        
        preds["GBM"].append(p_g)
        preds["Merton"].append(p_m)
        preds["Heston"].append(p_h)
        
        # Actualizar UI cada 10 pasos para no alentar
        if i % 5 == 0:
            progress_bar.progress((i + 1) / total)
            
    progress_bar.empty()
    return dates, preds

with st.spinner(f"Calibrando modelos con MLE/QMLE en ventana móvil ({TEST_DAYS} días)..."):
    dates_arr, predictions = run_rolling_forecast_oop(data, TEST_DAYS)

# -------------------------
# Métricas de Error (RMSE)
# -------------------------
test_prices = data.loc[dates_arr, "price"].values

rmse_gbm = np.sqrt(mean_squared_error(test_prices, predictions["GBM"]))
rmse_merton = np.sqrt(mean_squared_error(test_prices, predictions["Merton"]))
rmse_heston = np.sqrt(mean_squared_error(test_prices, predictions["Heston"]))

results_df = pd.DataFrame({
    "Modelo": ["Browniano (GBM)", "Merton", "Heston"],
    "RMSE": [rmse_gbm, rmse_merton, rmse_heston]
}).sort_values("RMSE").reset_index(drop=True)

best_model_name = results_df.loc[0, "Modelo"]

# -------------------------
# Predicción Final (Día Siguiente)
# -------------------------
# Usamos TODA la data histórica para la calibración final
all_prices = data["price"].values
#S_last = all_prices[-1]
S_last = float(data["price"].iloc[-1])

#S_last = float(S_last)


# Instanciar modelos finales con N_SIM alto
final_gbm = GBM()
final_gbm.fit(all_prices)
sims_gbm = final_gbm.simulate(S_last, N=N_SIM_FINAL)

final_mer = Merton()
final_mer.fit(all_prices)
sims_mer = final_mer.simulate(S_last, N=N_SIM_FINAL)

final_hes = Heston()
final_hes.fit(all_prices)
sims_hes = final_hes.simulate(S_last, N=N_SIM_FINAL) # Más steps

# Selección de simulaciones del ganador para percentiles
if "Browniano" in best_model_name:
    winner_sims = sims_gbm
    pred_val = np.mean(sims_gbm)
elif "Merton" in best_model_name:
    winner_sims = sims_mer
    pred_val = np.mean(sims_mer)
else:
    winner_sims = sims_hes
    pred_val = np.mean(sims_hes)

pct = np.percentile(winner_sims, [5, 25, 50, 75, 95])

# -------------------------
# DISPLAY: Tarjetas KPIs
# -------------------------
c1, c2, c3, c4 = st.columns(4)
c1.markdown(f"<div class='card'><div class='kpi-title'>Último Precio</div><div class='kpi-value'>${S_last:.2f}</div></div>", unsafe_allow_html=True)
c2.markdown(f"<div class='card'><div class='kpi-title'>Mejor Modelo</div><div class='kpi-value'>{best_model_name}</div><div class='kpi-sub'>RMSE: {results_df.loc[0,'RMSE']:.4f}</div></div>", unsafe_allow_html=True)
c3.markdown(f"<div class='card'><div class='kpi-title'>Predicción (T+1)</div><div class='kpi-value'>${pred_val:.2f}</div><div class='kpi-sub'>Esperanza matemática</div></div>", unsafe_allow_html=True)
c4.markdown(f"<div class='card'><div class='kpi-title'>Rango (90% Conf)</div><div class='kpi-value'>${pct[0]:.2f} - {pct[-1]:.2f}</div></div>", unsafe_allow_html=True)

st.markdown("---")

# -------------------------
# DISPLAY: Gráficos y Tablas
# -------------------------
col_L, col_R = st.columns([3, 2])

with col_L:
    st.markdown("<div class='section-title'>Validación Histórica (Rolling Backtest)</div>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(dates_arr, test_prices, label="Real", color="Black", linewidth=1.5, alpha=0.8)
    ax.plot(dates_arr, predictions["GBM"], label="GBM", linestyle="--", color="#f772f0")
    ax.plot(dates_arr, predictions["Merton"], label="Merton", linestyle="-.",color="#babd12")
    ax.plot(dates_arr, predictions["Heston"], label="Heston", linestyle=":",color="#00ff55")
    
    ax.set_facecolor("#f4f5f8")
    fig.patch.set_facecolor("#f9fafc")
    ax.tick_params(axis='x', colors='black')
    ax.tick_params(axis='y', colors='black')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.legend(facecolor="#121419", labelcolor="white")
    ax.grid(True, color="#333333", linestyle="--", alpha=0.5)
    st.pyplot(fig)

with col_R:
    st.markdown("<div class='section-title'>Ranking de Precisión</div>", unsafe_allow_html=True)
    st.dataframe(results_df.style.format({"RMSE": "{:.4f}"}), hide_index=True, use_container_width=True)
    
    st.markdown("<div class='section-title' style='margin-top:20px'>Distribución Probabilística</div>", unsafe_allow_html=True)
    df_pct = pd.DataFrame({
        "Percentil": ["5% (Pesimista)", "25%", "50% (Base)", "75%", "95% (Optimista)"],
        "Precio Estimado": pct
    })
    st.dataframe(df_pct.style.format({"Precio Estimado": "${:.2f}"}), hide_index=True, use_container_width=True)

    
