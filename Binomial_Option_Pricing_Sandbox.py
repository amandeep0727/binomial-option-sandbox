import streamlit as st
import numpy as np
import plotly.graph_objects as go

# ---------- Page setup
st.set_page_config(page_title="Option Pricing With Sensitivity Sandbox", layout="wide")
st.markdown("## Option Pricing Sandbox (Binomial)")

# ---------- Binomial pricer
class BinomialOptionPricer:
    def __init__(self, S0, K, T, r, sigma, N, c=0, option_type='call', american=False):
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.N = N
        self.c = c            # dividend yield / carry
        self.option_type = option_type.lower()
        self.american = american
        self._calculate_parameters()

    def _calculate_parameters(self):
        if self.N <= 0 or self.T <= 0:
            raise ValueError("N must be >=1 and T must be >0.")
        self.dt = self.T / self.N
        self.u = np.exp(self.sigma * np.sqrt(self.dt))
        self.d = np.exp(-self.sigma * np.sqrt(self.dt))
        self.q = (np.exp((self.r - self.c) * self.dt) - self.d) / (self.u - self.d)
        if not (0 < self.q < 1):
            raise ValueError(f"Risk-neutral probability out of bounds: q={self.q:.4f}. "
                             "Increase N or adjust inputs.")
        self.discount_factor = np.exp(-self.r * self.dt)

    def build_lattices(self):
        N = self.N
        stock  = np.full((N+1, N+1), np.nan)
        option = np.full((N+1, N+1), np.nan)
        early  = np.full((N+1, N+1), '', dtype=object)

        # Stock lattice
        for t in range(N+1):
            i = np.arange(t+1)
            stock[:t+1, t] = self.S0 * (self.u ** i) * (self.d ** (t - i))

        # Terminal payoff
        if self.option_type == 'call':
            option[:, N] = np.maximum(stock[:, N] - self.K, 0.0)
        else:
            option[:, N] = np.maximum(self.K - stock[:, N], 0.0)

        # Backward induction
        for t in range(N-1, -1, -1):
            for i in range(t+1):
                hold = (self.q * option[i+1, t+1] + (1 - self.q) * option[i, t+1]) * self.discount_factor
                if self.american:
                    exercise = (stock[i, t] - self.K) if self.option_type == 'call' else (self.K - stock[i, t])
                    exercise = max(exercise, 0.0)
                    option[i, t] = max(hold, exercise)
                    early[i, t] = 'Yes' if exercise > hold else ''
                else:
                    option[i, t] = hold

        self.stock_lattice  = stock
        self.option_lattice = option
        self.early_exercise = early
        return option[0, 0]

def root_greeks(pr):
    """Delta/Gamma at t=0 from lattice nodes."""
    S0, u, d = pr.S0, pr.u, pr.d
    V = pr.option_lattice
    Vu, Vd = V[1, 1], V[0, 1]
    Su, Sd = S0 * u, S0 * d
    delta = (Vu - Vd) / (Su - Sd)

    Vuu, Vud, Vdd = V[2, 2], V[1, 2], V[0, 2]
    Suu, Sud, Sdd = S0 * (u ** 2), S0, S0 * (d ** 2)
    gamma = ((Vuu - Vud) / (Suu - Sud) - (Vud - Vdd) / (Sud - Sdd)) / ((Suu - Sdd) / 2)
    return float(delta), float(gamma)

def price_with(S0, K, T, r, sigma, q, N, opt_type, american):
    pr_local = BinomialOptionPricer(S0=S0, K=K, T=T, r=r, sigma=sigma, N=N,
                                    c=q, option_type=opt_type, american=american)
    return pr_local.build_lattices()

# --- NEW: data utilities ---
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

TOP10 = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "BRK-B", "LLY", "AVGO", "JPM"]

@st.cache_data(show_spinner=False, ttl=600)
def fetch_1y_history(ticker: str):
    end = datetime.today().date()
    start = end - timedelta(days=400)  # pad a bit for holidays
    # yfinance uses '-' not '.' in BRK-B
    yf_ticker = ticker.replace(".", "-")
    data = yf.download(yf_ticker, start=start, end=end, auto_adjust=True, progress=False)
    # dividends (cash) as a separate call; some tickers return in actions
    try:
        div = yf.Ticker(yf_ticker).dividends
    except Exception:
        div = pd.Series(dtype="float64")
    return data, div

def estimate_inputs_from_data(price_df: pd.DataFrame, div_series: pd.Series):
    # Spot from last available Close
    S0_est = float(price_df["Close"].dropna().iloc[-1])

    # Realized vol from daily log returns
    px = price_df["Close"].dropna()
    rets = np.log(px/px.shift(1)).dropna()
    sigma_est = float(rets.std() * np.sqrt(252))
    # guard against degenerate series
    if not np.isfinite(sigma_est) or sigma_est <= 0:
        sigma_est = 0.20  # fallback

    # Dividend yield ~ trailing 1y cash / Spot
    if div_series is not None and len(div_series) > 0:
        div_1y = div_series[div_series.index >= (div_series.index.max() - pd.Timedelta(days=365))]
        total_cash = float(div_1y.sum()) if len(div_1y) else 0.0
        q_est = float(total_cash / S0_est) if S0_est > 0 else 0.0
    else:
        q_est = 0.0

    return S0_est, sigma_est, q_est


# ---------- Layout: left controls | right outputs
left, right = st.columns([1, 2], gap="large")

with left:
    st.markdown("**Data source**")
    ticker = st.selectbox("Ticker", TOP10, index=0)
    fetch_now = st.button("Fetch 1-year data", use_container_width=True)

    # Defaults (before fetch)
    S0_default, sigma_default, q_default = 100.0, 0.20, 0.00

    if fetch_now:
        with st.spinner(f"Downloading data for {ticker}…"):
            df, div = fetch_1y_history(ticker)
        if df.empty:
            st.warning("No price data found. Using defaults.")
            S0_prefill, sigma_prefill, q_prefill = S0_default, sigma_default, q_default
        else:
            S0_prefill, sigma_prefill, q_prefill = estimate_inputs_from_data(df, div)
            st.caption(f"S0={S0_prefill:.2f}, σ={sigma_prefill:.3f}, q={q_prefill:.3f} (from last 1y).")
    else:
        S0_prefill, sigma_prefill, q_prefill = S0_default, sigma_default, q_default

    st.markdown("---")
    st.markdown("**Model inputs**")
    S0 = st.slider("Spot S0", 10.0, 1000.0, float(S0_prefill), 1.0)
    K  = st.slider("Strike K", 10.0, 1000.0, float(np.clip(S0_prefill, 10, 1000)), 1.0)
    T  = st.slider("Maturity T (yrs)", 0.05, 5.0, 1.0, 0.05)
    r  = st.slider("Rate r", -0.02, 0.15, 0.04, 0.005)  # default ~4%
    q  = st.slider("Dividend yield q", 0.0, 0.10, float(min(q_prefill, 0.10)), 0.001)
    sigma = st.slider("Vol σ", 0.05, 1.00, float(np.clip(sigma_prefill, 0.05, 1.00)), 0.005)
    N  = st.slider("Steps N", 25, 600, 150, 5)
    opt_type = st.selectbox("Type", ["call", "put"], index=0)




# top-right toggle row
with right:
    hdr1, hdr2, hdr3, hdr4 = st.columns([1, 1, 1, 0.9])
    with hdr4:
        american = st.toggle("American", value=True, help="Price as American option")

# ---------- Compute base price & render main curve
try:
    pr = BinomialOptionPricer(S0, K, T, r, sigma, N, c=q, option_type=opt_type, american=american)
    price = pr.build_lattices()
    delta, gamma = root_greeks(pr)

    with right:
        # --- Metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Price", f"{price:,.4f}")
        m2.metric("Delta", f"{delta:,.4f}")
        m3.metric("Gamma", f"{gamma:,.6f}")

        # --- Value vs Spot curve
        S_grid = np.linspace(0.6 * S0, 1.4 * S0, 121)
        vals = []
        for s in S_grid:
            vals.append(price_with(s, K, T, r, sigma, q, N, opt_type, american))

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=S_grid, y=vals, mode="lines", name="Price"))
        fig.add_vline(x=S0, line_dash="dash", opacity=0.85)
        mode_label = "American" if american else "European"
        fig.update_layout(
            template="plotly_dark",
            height=520,
            margin=dict(l=40, r=30, t=60, b=40),
            title=f"Option value vs Spot | {opt_type.title()} | {mode_label}",
            xaxis_title="Spot S",
            yaxis_title="Option price",
            xaxis=dict(ticks="outside", showgrid=True, zeroline=False),
            yaxis=dict(ticks="outside", showgrid=True, zeroline=False),
        )
        st.plotly_chart(fig, use_container_width=True)

        if american:
            st.caption("Tip: Toggle American off to see the value of early-exercise optionality.")

        # ---------------- Sensitivity (Tornado) ----------------
        st.markdown("### Sensitivity (Tornado)")

        colA, colB = st.columns([1.2, 1])
        with colA:
            shock_pct = st.slider("Shock size (% of parameter)", 1, 20, 10, 1) / 100.0
        with colB:
            unit = st.radio("Show impact as", ["Absolute (Δ price)", "% of price"], index=0, horizontal=True)

        # Sensible absolute floors to keep shocks meaningful and positive
        dS  = max(0.01 * S0, shock_pct * S0)
        dK  = max(0.01 * K,  shock_pct * K)
        ds  = max(0.005,     shock_pct * sigma)     # σ floor
        dT  = max(1/365,     shock_pct * T)         # at least 1 day
        dr  = max(0.0025,    shock_pct * max(0.01, abs(r)))  # 25 bps or scaled
        dq  = max(0.0025,    shock_pct * max(0.005, q + 1e-9))

        params = [
            ("Spot (S₀)", "S0", dS),
            ("Strike (K)", "K", dK),
            ("Volatility (σ)", "sigma", ds),
            ("Maturity (T)", "T", dT),
            ("Rate (r)", "r", dr),
            ("Dividend (q)", "q", dq),
        ]

        impacts = []
        labels  = []
        base = float(price)

        # helper to clamp positive parameters
        def clamp_pos(x, floor):
            return max(floor, x)

        for label, key, d in params:
            S0p, Kp, Tp, rp, sigmap, qp = S0, K, T, r, sigma, q
            if key == "S0":
                high = price_with(S0p + d, Kp, Tp, rp, sigmap, qp, N, opt_type, american)
                low  = price_with(clamp_pos(S0p - d, 1e-6), Kp, Tp, rp, sigmap, qp, N, opt_type, american)
            elif key == "K":
                high = price_with(S0p, Kp + d, Tp, rp, sigmap, qp, N, opt_type, american)
                low  = price_with(S0p, clamp_pos(Kp - d, 1e-6), Tp, rp, sigmap, qp, N, opt_type, american)
            elif key == "sigma":
                high = price_with(S0p, Kp, Tp, rp, clamp_pos(sigmap + d, 1e-6), qp, N, opt_type, american)
                low  = price_with(S0p, Kp, Tp, rp, clamp_pos(sigmap - d, 1e-6), qp, N, opt_type, american)
            elif key == "T":
                high = price_with(S0p, Kp, clamp_pos(Tp + d, 1e-6), rp, sigmap, qp, N, opt_type, american)
                low  = price_with(S0p, Kp, clamp_pos(Tp - d, 1e-6), rp, sigmap, qp, N, opt_type, american)
            elif key == "r":
                high = price_with(S0p, Kp, Tp, rp + d, sigmap, qp, N, opt_type, american)
                low  = price_with(S0p, Kp, Tp, rp - d, sigmap, qp, N, opt_type, american)
            elif key == "q":
                high = price_with(S0p, Kp, Tp, rp, sigmap, clamp_pos(qp + d, 0.0), N, opt_type, american)
                low  = price_with(S0p, Kp, Tp, rp, sigmap, clamp_pos(qp - d, 0.0), N, opt_type, american)
            else:
                continue

            # impact as the larger of |high-base| and |low-base|
            impact_abs = float(max(abs(high - base), abs(low - base)))
            if unit.startswith("%") and base != 0:
                impact = 100.0 * impact_abs / abs(base)
            else:
                impact = impact_abs

            impacts.append(impact)
            labels.append(label)

        # sort bars by impact desc
        order = np.argsort(impacts)[::-1]
        impacts_sorted = [impacts[i] for i in order]
        labels_sorted  = [labels[i]  for i in order]

        bar_title = "Option Value Sensitivity (Tornado) — " + ("%" if unit.startswith("%") else "Δ price")

        fig_t = go.Figure()
        fig_t.add_trace(go.Bar(
            x=impacts_sorted,
            y=labels_sorted,
            orientation="h",
            marker=dict(line=dict(width=0)),
            name="Impact"
        ))
        fig_t.update_layout(
            template="plotly_dark",
            height=420,
            margin=dict(l=80, r=30, t=50, b=40),
            title=bar_title,
            xaxis_title=("Impact (% of price)" if unit.startswith("%") else "Impact (price units)"),
            yaxis_title="Parameter",
        )
        st.plotly_chart(fig_t, use_container_width=True)

except ValueError as e:
    st.error(str(e))
