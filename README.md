# Binomial Option Pricing Sandbox

An interactive **Streamlit app** that prices **European and American options** using the **Binomial Lattice model** with real-time Greeks and sensitivity analysis.

Built as part of my **Financial Engineering and Risk Management** learning journey—bridging quantitative theory with production-grade implementation.

---

## 🎯 Key Features

### **Core Pricing Engine**
- Binomial lattice model supporting both **European** and **American** options
- Supports **Call** and **Put** options
- Real-time pricing with adjustable parameters
- High-performance vectorized NumPy computations

### **Interactive Parameters**
- **Spot Price (S₀):** Current asset price
- **Strike (K):** Exercise price
- **Maturity (T):** Time to expiration (years)
- **Risk-free Rate (r):** Discount rate
- **Dividend Yield (q):** Continuous dividend payment
- **Volatility (σ):** Annualized standard deviation
- **Steps (N):** Lattice depth (25–600 steps for precision)

### **Greeks & Risk Metrics**
- **Delta:** Option sensitivity to spot price changes
- **Gamma:** Delta sensitivity (convexity)
- Computed at t=0 from lattice nodes for precision

### **Real-Time Visualizations**
- **Option Value vs Spot Price:** Interactive Plotly curve showing payoff structure and Greeks impact
- **Sensitivity Analysis (Tornado):** Parameter impact ranking—identify key value drivers
- Supports absolute (Δ price) and percentage (% of price) impact views

### **Market Data Integration**
- Fetch **1-year historical data** for 10 major equities (AAPL, MSFT, NVDA, AMZN, etc.)
- Auto-estimate model inputs from data:
  - Spot price from latest close
  - Volatility from realized daily log returns (annualized)
  - Dividend yield from historical dividend data
- Streamlines setup for real-world pricing scenarios

---

## 🛠️ Tech Stack

- **Streamlit** — Interactive dashboard framework
- **Plotly** — Professional-grade visualizations (value curves, tornado charts)
- **NumPy** — Vectorized numerical computing
- **Pandas** — Data handling and time-series analysis
- **yfinance** — Market data retrieval

---

## 🚀 Getting Started

### **Installation**

```bash
git clone https://github.com/amandeep0727/binomial-option-sandbox.git
cd binomial-option-sandbox
pip install -r requirements.txt
```

### **Running Locally**

```bash
streamlit run Binomial_Option_Pricing_Sandbox.py
```

Then visit `http://localhost:8501` in your browser.

### **Live Demo**

[👉 Access the Live App on Streamlit Cloud](https://binomial-pricer.streamlit.app)

---

## 📋 Model Methodology

### **Binomial Lattice Construction**
The model builds a recombining lattice over N steps:
- **Up-move factor:** \(u = e^{\sigma\sqrt{\Delta t}}\)
- **Down-move factor:** \(d = e^{-\sigma\sqrt{\Delta t}}\)
- **Risk-neutral probability:** \(q = \frac{e^{(r-q)\Delta t} - d}{u - d}\)

### **Terminal Payoff**
At expiration (t=T):
- **Call:** \(\max(S_T - K, 0)\)
- **Put:** \(\max(K - S_T, 0)\)

### **Backward Induction**
Starting from terminal nodes, work backward to t=0:
- **European:** \(V_t = e^{-r\Delta t}[q \cdot V_{u,t+1} + (1-q) \cdot V_{d,t+1}]\)
- **American:** \(V_t = \max(\text{intrinsic value}, \text{hold value})\)

### **Greeks Calculation**
Computed from lattice nodes at t=0:
- **Delta:** \(\Delta = \frac{V_{u,1} - V_{d,1}}{S_u - S_d}\)
- **Gamma:** Second derivative approximation from three nodes

---

## 📊 Use Cases

✅ **Quantitative Finance Education** — Understand binomial models visually  
✅ **Option Trading Strategy Design** — Evaluate payoff structures and sensitivities  
✅ **Risk Assessment** — Analyze Greeks and parameter impacts on option value  
✅ **Portfolio Management** — Price derivatives for hedging analysis  
✅ **Interview Preparation** — Demonstrate production-grade quantitative skills  

---

## 🔬 Advanced Features

- **Sensitivity Analysis (Tornado):** Rank parameters by impact on option value
- **Real-time Parameter Adjustment:** Instant updates to pricing and Greeks
- **Market Data Integration:** Estimate implied volatility and dividend yields from historical data
- **Production-Grade Code:** Modular architecture, error handling, and numerical stability checks
- **Scalable Design:** Efficient for high-step lattices (up to 600 steps)

---

## 📝 Code Structure

```
binomial-option-sandbox/
├── Binomial_Option_Pricing_Sandbox.py   # Main Streamlit app
├── requirements.txt                      # Dependencies
└── README.md                             # This file
```

### **Key Components**

- **BinomialOptionPricer class** — Core pricing engine with lattice construction
- **root_greeks()** — Greeks calculation at t=0
- **price_with()** — Utility for sensitivity analysis
- **fetch_1y_history()** — Market data retrieval with caching
- **Streamlit UI** — Left sidebar (inputs) | Right panel (outputs)

---

## 📚 References & Learning

This project implements concepts from:
- **Columbia Financial Engineering & Risk Management** — Derivatives Pricing
- **Hull, J.C. (2017)** — Options, Futures, and Other Derivatives
- **Black-Scholes-Merton Model** — Continuous-time option pricing foundations

---

## 🎓 Learning Outcomes

Through this project, I demonstrated:
- ✅ **Quantitative Modeling:** Binomial lattice construction and backward induction
- ✅ **Python Engineering:** Vectorized NumPy, Streamlit deployment, production-grade code
- ✅ **Risk Analytics:** Greeks calculations and sensitivity analysis
- ✅ **UI/UX Design:** Interactive Plotly visualizations and parameter controls
- ✅ **Real-World Application:** Market data integration and practical valuation

---

## 🤝 Contributing

Feedback and suggestions are welcome! Open an issue or submit a pull request.

---

## 📄 License

MIT License — See LICENSE file for details

---

## 👨‍💻 Author

**Amandeep Singh**  
Financial Engineering & Data Science | Quantitative Finance Enthusiast  
[GitHub](https://github.com/amandeep0727) | [LinkedIn](https://linkedin.com/in/amandeep-singh-strategy)

---

## 🔗 Related Projects

- **Portfolio Optimization Engine** — Mean-variance CVaR optimization with Monte Carlo validation
- **Python Options Pricing** — Black-Scholes & Monte Carlo implementations with Greeks
- **DCC-GARCH Risk Model** — Multivariate volatility forecasting and tail-risk assessment

---

_Last Updated: October 2025_