# 📈 Alpha-Geeks: Indian Options Analytics & ML Engine

**Alpha-Geeks** is a comprehensive quantitative finance tool designed specifically for the **Indian Derivatives Market (NSE)**. It combines advanced Black-Scholes pricing, interactive 3D volatility visualizations, and Machine Learning algorithms to analyze option sensitivities (Greeks) and detect market volatility regimes.

## 🚀 Features

### 1. 📊 Advanced Greeks Calculator
- Real-time calculation of **Delta, Gamma, Vega, Theta, and Rho**.
- Support for **Nifty, Bank Nifty**, and individual NSE stocks.
- Numerical validation of Greeks for high-precision modeling.

### 2. 🧊 Interactive 3D Visualization
- **3D Volatility Surfaces**: visualize how option prices change with Spot Price and Time to Expiry.
- **Heatmaps**: Analyze Greek sensitivities across different strike prices and expiries.
- Built using **Plotly** for fully interactive rotation and zooming.

### 3. 🤖 ML Volatility Regime Detection
- Implements **Random Forest Classifiers** to detect market regimes (High Volatility vs. Low Volatility).
- Engineers features from historical price data to predict future volatility states.
- Provides classification metrics (Accuracy, ROC-AUC) and Feature Importance charts.

### 4. 💼 Multi-Leg Strategy Builder
- Pre-built templates for complex strategies:
  - **Straddles & Strangles** (Long/Short)
  - **Iron Condors**
  - **Bull/Bear Spreads**
- **Risk Metrics**: Net Delta, Net Vega, and Max Profit/Loss scenarios.

### 5. 📉 P&L Risk Simulation
- Simulates portfolio performance under various "What-If" scenarios.
- Analyze impact of **Spot Price shifts** (±20%) and **Volatility shocks** on your strategies.

---

## 🛠️ Tech Stack

* **Core Logic:** Python, NumPy, Pandas
* **Pricing Models:** Black-Scholes-Merton (BSM)
* **Machine Learning:** Scikit-Learn (Random Forest)
* **Visualization:** Plotly, Matplotlib
* **Frontend:** Streamlit

---

## 💻 Installation & Setup

Follow these steps to run the application locally:

**1. Clone the Repository**
```bash
git clone [https://github.com/Saimohith2402/Alpha-Geeks.git](https://github.com/Saimohith2402/Alpha-Geeks.git)
cd Alpha-Geeks
2. Create a Virtual EnvironmentBash# Windows
python -m venv venv
.\venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
3. Install DependenciesBashpip install -r requirements.txt
4. Run the ApplicationNavigate to the app folder (or wherever the main file is located) and run:Bashstreamlit run indian_streamlit_app.py
```

Author: 
Daggolu Sai Mohith Reddy
Mechanical Engineering Undergrad
IIT Ropar 

