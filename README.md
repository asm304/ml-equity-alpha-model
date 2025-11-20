# 📈 ML Equity Alpha Model

A machine-learning equity alpha model using **Ridge Regression** and **XGBoost**, with walk-forward cross-validation, z-score portfolio construction, and long/short backtesting over the **Russell 3000** universe.

---

## 🚀 Overview

This project builds a **cross-sectional equity return prediction model** and evaluates it using a realistic walk-forward investment simulation.

The workflow mimics how a quantitative equity team develops stock-selection signals:

- Data ingestion + feature engineering  
- Target creation (next-month returns)  
- Model training (Ridge, XGBoost)  
- Walk-forward cross-validation  
- Portfolio construction (z-score weights)  
- Backtesting (long-only, short-only, long–short)  
- Performance analytics (returns, Sharpe, drawdowns, turnover)

All modeling uses only **past information** at each point in time (no look-ahead bias).

---

## 🧠 Methodology

### **1. Universe & Target**

- **Universe:** Russell 3000  
- **Frequency:** Monthly  

**Target variable (next-month total return):**

\[
r_{i,t+1} = \frac{P_{i,t+1}}{P_{i,t}} - 1
\]

---

### **2. Features**

The model uses a diverse set of cross-sectional predictors:

- Price-based factors (momentum, volatility, reversals)  
- Fundamental ratios  
- Rolling statistical features  
- Sector/dummy encodings  
- Technical indicators  

All features are **lagged** so only past information is used.

---

### **3. Models Tested**

#### ✅ **Ridge Regression (best performer)**  
- Stable  
- Interpretable  
- Resistant to multicollinearity  
- Strong information coefficient stability over time  

#### ⚠️ **XGBoost**
- Useful for nonlinear relationships  
- Higher variance  
- Underperformed Ridge on this dataset  

➡️ **Final backtest uses Ridge as the primary model.**

---

### **4. Walk-Forward Training**

For each month **t**:

1. Use the prior **36 months** as the training window  
2. Fit the model on that window  
3. Predict returns for month **t+1**  
4. Move forward one month  
5. Repeat  

This replicates real quant workflows and prevents information leakage.

---

### **5. Portfolio Construction — Z-Score Weights**

For each month’s predicted returns:

1. Select top and/or bottom scoring stocks (based on mode)  
2. Standardize via z-scores  
3. Clip extreme values (default: ±3)  
4. Normalize weights to sum to 1  

**Weighting modes:**

- **L** → Long-only  
- **S** → Short-only  
- **LS** → Long–Short (long top bucket, short bottom bucket)

---

## 📊 Performance Summary  
*(2018-01-31 → 2025-10-31)*

### **Long-Only Portfolio (Best Performer)**  
- **Annual Return:** 1.148  
- **Annual Volatility:** 1.49  
- **Sharpe:** 0.77  
- **Avg Turnover:** 0.41  
- **Max Drawdown:** –1.096  

**Interpretation:**  
Strong alpha on the long side — model is excellent at identifying winners.

---

### **Short-Only Portfolio**  
- **Annual Return:** –0.378  
- **Sharpe:** –0.59  
- **Max Drawdown:** –2.945  

**Interpretation:**  
Model struggles to identify losers — common in equity ML due to noise, crowding, and mean-reversion.

---

### **Long–Short Portfolio**  
- **Annual Return:** 0.395  
- **Sharpe:** 0.80  
- **Max Drawdown:** –0.271  

Hedging reduces volatility, but performance is diluted because the short leg loses money.

---

## 🧩 Key Insights

- **Predictive asymmetry:**  
  The signal is better at finding winners than losers — typical of equity ML.

- **Ridge > XGBoost:**  
  Linear regularized models outperform tree models in noisy cross-sectional datasets.

- **Turnover is realistic:**  
  ~0.41/month is typical for quant equity factors.

- **Alpha is concentrated in the long leg:**  
  Short predictions behave differently (crowding, noise, microstructure effects).

---

## 🛠️ Future Enhancements

- Add richer fundamental datasets  
- Try beta-neutral or sector-neutral portfolio construction  
- PCA / autoencoder factor denoising  
- Experiment with neural nets (TabNet, MLP, GNN)  
- Add risk-model overlays (Barra-style)  

---

## ▶️ How to Run

```bash
pip install -r requirements.txt
jupyter notebook
```


Run the notebooks in order:

01_data_prep.ipynb

02_feature_engineering.ipynb

03_modeling.ipynb

04_backtest.ipynb

05_report.ipynb (optional summary/plots)

---

## 📬 Contact

Created by Aman Meda
