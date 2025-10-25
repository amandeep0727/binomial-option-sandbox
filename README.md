# Binomial Option Pricing Sandbox

An interactive **Streamlit app** that prices **European and American options** using the **Binomial Lattice model**.

Built as part of my Financial Engineering and Risk Management learning journey — bridging quantitative theory with real-time implementation.

---

## 🎯 Features

- Adjustable parameters:
  - Spot price (S₀)
  - Strike (K)
  - Maturity (T)
  - Risk-free rate (r)
  - Dividend yield (q)
  - Volatility (σ)
  - Steps (N)
- Switch between **Call / Put** and **American / European**
- Real-time **option price, Delta, and Gamma**
- Interactive Plotly visualization: *Option Value vs Spot Price*

---

## ⚙️ Tech Stack

- **Streamlit** — interactive dashboard  
- **Plotly** — professional-grade visualization  
- **NumPy** — numerical calculations  

---

## 🚀 Running Locally

```bash
git clone https://github.com/amandeep0727/binomial-option-sandbox.git
cd binomial-option-sandbox
pip install -r requirements.txt
streamlit run Binomial_Option_Pricing_Sandbox.py