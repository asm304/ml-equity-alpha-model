📈 ML Equity Alpha Model

A machine-learning equity alpha model using Ridge and XGBoost with walk-forward cross-validation, z-score portfolio construction, and long/short backtesting over the Russell 3000 universe.

🚀 Overview

This project builds a cross-sectional equity return prediction model using machine learning and evaluates it in a realistic walk-forward investment simulation.

The workflow replicates how a quantitative equity team develops and evaluates stock-ranking signals:

Data ingestion + feature engineering

Target creation (next-month returns)

Model training (Ridge, XGBoost)

Walk-forward cross-validation

Portfolio construction via z-score weighting

Backtesting long-only, short-only, and long-short portfolios

Performance analytics (returns, Sharpe, drawdowns, turnover)

All modeling uses only past information at each point in time, avoiding look-ahead bias.

🗂 Repository Structure
ml-equity-alpha-model/
│
├── 01_data_prep.ipynb         # Data loading, cleaning, feature creation
├── 02_feature_engineering.ipynb
├── 03_modeling.ipynb          # Ridge + XGBoost training and predictions
├── 04_backtest.ipynb          # Z-score weighting and portfolio simulation
├── 05_report.ipynb            # Summary charts, stats, and final analysis
│
├── data/                      # Intermediate processed data (ignored in Git)
├── README.md
├── .gitignore
└── requirements.txt (optional)

🧪 Methodology
1. Universe & Target

Universe: Russell 3000 constituents

Frequency: Monthly

Target variable:

𝑟
𝑖
,
𝑡
+
1
=
𝑃
𝑖
,
𝑡
+
1
𝑃
𝑖
,
𝑡
−
1
r
i,t+1
	​

=
P
i,t
	​

P
i,t+1
	​

	​

−1

(Next-month total return)

2. Features

The model uses a diverse set of cross-sectional predictors such as:

Price-based factors (momentum, volatility, reversals)

Fundamental ratios

Rolling statistical features

Sector/dummy encodings

Technical signals

All features are lagged so only past information is used.

3. Models

Two ML models were tested:

✅ Ridge Regression (best performer)

Stable

Interpretable

Resistant to multicollinearity

Strong IC stability across time

⚠️ XGBoost

Useful for nonlinear relationships

Higher variance

Performed worse cross-sectionally than Ridge in this dataset

Final backtest uses Ridge as the primary model.

4. Walk-Forward Training

For each month t:

Use the prior 36 months of data as the training window

Fit the model only on that period

Predict stock returns for month t+1

Move window forward one month

Repeat

This mimics real quant workflows and prevents information leakage.

5. Portfolio Construction — Z-Score Weights

For each month’s predicted returns:

Select the highest / lowest scoring stocks

Standardize via z-scores

Clip extreme values to avoid concentration

Normalize weights

Weighting modes:

Mode	Description
L	Long-Only: invest in top-ranked stocks only
S	Short-Only: hold inverse positions in bottom-ranked stocks
LS	Long-Short: go long the top bucket and short the bottom bucket
📊 Performance Summary (2018-01-31 → 2025-10-31)
Long-Only Portfolio (best approach)

Annual return: 1.148
Annual vol: 1.49
Sharpe: 0.77
Avg turnover: 0.41
Max drawdown: –1.096

The model demonstrates strong predictive ability on the long side, producing a high Sharpe and consistent alpha.

Short-Only Portfolio

Annual return: –0.378
Sharpe: –0.59
Max drawdown: –2.945

The short leg underperforms materially. Cross-sectional signals often have asymmetric predictive power, and this model is significantly better at finding winners than losers.

Long-Short Portfolio

Annual return: 0.395
Sharpe: 0.80
Max drawdown: –0.271

Volatility decreases from hedging, so Sharpe looks healthy, but performance is diluted because the short side loses money. This confirms that the model’s alpha is concentrated in the long leg.

🧠 Key Insights
✔ Predictive asymmetry

The model is good at predicting outperformers, but not underperformers.
This is common in equity ML where short returns behave differently (crowding, mean-reversion, microstructure noise).

✔ Walk-forward CV prevents overfitting

Each month is predicted out-of-sample.

✔ Ridge > XGBoost

Linear, regularized models often outperform tree-based models in noisy cross-sectional equity datasets.

✔ Turnover is reasonable

A turnover of ~0.41/month is typical for quant stock selection strategies.

📌 Future Improvements

Add richer fundamental datasets

Try sector-neutral or beta-neutral portfolio construction

Add PCA-denoised factor features

Test neural nets (TabNet, MLP, GNN)

Incorporate risk-model overlays (Barra-style)

🛠 How to Run

(Assuming Jupyter + Python 3.10+)

pip install -r requirements.txt
jupyter notebook


Run notebooks in order:

1 → 2 → 3 → 4 → 5

📬 Contact

Created by Aman Meda