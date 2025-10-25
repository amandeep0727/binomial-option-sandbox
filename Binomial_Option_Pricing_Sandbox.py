

# Binomial_Option_Pricing_Sandbox.py
import streamlit as st
import numpy as np
import plotly.graph_objects as go

# ---------- Page setup
st.set_page_config(page_title="Option Pricing Sandbox", layout="wide")
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

# ---------- Layout: left controls | right outputs
left, right = st.columns([1, 2], gap="large")

with left:
    S0 = st.slider("Spot S0", 10.0, 500.0, 100.0, 1.0)
    K  = st.slider("Strike K", 10.0, 500.0, 100.0, 1.0)
    T  = st.slider("Maturity T (yrs)", 0.05, 5.0, 1.0, 0.05)
    r  = st.slider("Rate r", -0.02, 0.15, 0.05, 0.005)
    q  = st.slider("Dividend yield q", 0.0, 0.10, 0.00, 0.005)
    sigma = st.slider("Vol σ", 0.05, 1.00, 0.20, 0.01)
    N  = st.slider("Steps N", 25, 400, 100, 5)
    opt_type = st.selectbox("Type", ["call", "put"], index=0)

# top-right toggle row
with right:
    hdr1, hdr2, hdr3, hdr4 = st.columns([1, 1, 1, 0.9])
    with hdr4:
        american = st.toggle("American", value=True, help="Price as American option")

# ---------- Compute & render
try:
    pr = BinomialOptionPricer(S0, K, T, r, sigma, N, c=q, option_type=opt_type, american=american)
    price = pr.build_lattices()
    delta, gamma = root_greeks(pr)

    with right:
        m1, m2, m3 = st.columns(3)
        m1.metric("Price", f"{price:,.4f}")
        m2.metric("Delta", f"{delta:,.4f}")
        m3.metric("Gamma", f"{gamma:,.6f}")

        # Value vs Spot curve
        S_grid = np.linspace(0.6 * S0, 1.4 * S0, 121)
        vals = []
        for s in S_grid:
            p2 = BinomialOptionPricer(s, K, T, r, sigma, N, c=q, option_type=opt_type, american=american)
            vals.append(p2.build_lattices())

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

except ValueError as e:
    st.error(str(e))
