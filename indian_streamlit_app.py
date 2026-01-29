import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from indian_bs_model import IndianBlackScholesModel, INDIAN_INDICES
from indian_portfolio_simulator import IndianOptionPortfolio, IndianOptionLeg, IndianStrategyBuilder
from indian_data_loader import IndianMarketDataLoader, IndianDataHelper
from indian_visualization_interactive_3d import IndianGreeksInteractiveVisualizer
from indian_ml_regime_model import IndianVolatilityRegimeDetector

st.set_page_config(
    page_title="Indian Options Greeks Calculator",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #FF6B00;
        text-align: center;
        padding: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🇮🇳 Indian Options Greeks Calculator + ML Regime Detection</div>', 
            unsafe_allow_html=True)
st.markdown('<p style="text-align:center">Greeks Analysis • Interactive 3D Visualization • ML Volatility Regime • Multi-Leg Strategies</p>', 
            unsafe_allow_html=True)
st.markdown("---")

if 'portfolio' not in st.session_state:
    st.session_state.portfolio = IndianOptionPortfolio()

if 'data_loader' not in st.session_state:
    st.session_state.data_loader = IndianMarketDataLoader()

if 'ml_detector' not in st.session_state:
    st.session_state.ml_detector = None

st.sidebar.header("📊 Indian Market Selector")

instrument_type = st.sidebar.radio("Choose Instrument Type", ["Index", "Stock"])

if instrument_type == "Index":
    index_list = list(INDIAN_INDICES.keys())
    selected_symbol = st.sidebar.selectbox("Select Index", index_list)
    symbol_display = INDIAN_INDICES[selected_symbol]['name']
else:
    stock_symbol = st.sidebar.text_input("Enter Stock Symbol (e.g., TCS, RELIANCE)")
    selected_symbol = stock_symbol.upper() if stock_symbol else "TCS"
    symbol_display = selected_symbol

st.sidebar.subheader(f"💰 {symbol_display}")
live_data = st.session_state.data_loader.get_live_price(selected_symbol)

if 'error' not in live_data:
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric("Price", f"₹{live_data['price']}")
    with col2:
        change_color = "🟢" if live_data['change'] >= 0 else "🔴"
        st.metric("Change", f"{change_color} {live_data['change_pct']:+.2f}%")
    
    spot_price = live_data['price']
    historical_vol = st.session_state.data_loader.get_historical_volatility(selected_symbol)
else:
    st.sidebar.warning(f"Could not fetch live data. Using default values.")
    spot_price = 100
    historical_vol = 0.20

st.sidebar.subheader("📅 Expiry Selection")
expiry_options = st.session_state.data_loader.get_expiry_days_list(selected_symbol)
selected_expiry = st.sidebar.selectbox("Select Expiry Type", list(expiry_options.keys()))
days_to_expiry = expiry_options[selected_expiry]

st.sidebar.subheader("⚙️ Greeks Parameters")
risk_free_rate = st.sidebar.slider("Risk-Free Rate", 0.01, 0.15, 0.065, 0.005)
volatility = st.sidebar.slider("Volatility (σ)", 0.05, 1.0, historical_vol, 0.01)

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Single Option Greeks",
    "3️⃣ Interactive 3D Visualization",
    "💼 Multi-Leg Strategies", 
    "🎯 Strategy Builder",
    "📊 P&L Analysis",
    "🤖 ML Regime Detection"
])

with tab1:
    st.header(f"Single Option Greeks - {symbol_display}")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        strike = st.number_input("Strike Price (₹)", value=spot_price, step=10)
    with col2:
        option_type = st.selectbox("Option Type", ["Call", "Put"])
    with col3:
        st.write("")
        st.write("")
        calculate = st.button("🔄 Calculate Greeks", type="primary", use_container_width=True)
    
    if calculate:
        try:
            model = IndianBlackScholesModel(
                S=spot_price,
                K=strike,
                days_to_expiry=days_to_expiry,
                r=risk_free_rate,
                sigma=volatility,
                option_type=option_type.lower()
            )
            
            greeks = model.all_greeks()
            
            st.success("✅ Greeks Calculated Successfully!")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("📍 Current Price", f"₹{greeks['price']:.2f}")
                st.metric("📊 Delta", f"{greeks['delta']:.4f}",
                         help=f"Move ₹1 → Option changes ₹{greeks['delta']:.4f}")
            
            with col2:
                st.metric("📈 Gamma", f"{greeks['gamma']:.6f}",
                         help="Delta change per ₹1 move")
                st.metric("🌊 Vega", f"{greeks['vega']:.4f}",
                         help="Change per 1% volatility increase")
            
            with col3:
                st.metric("⏳ Theta/Day", f"₹{greeks['theta']:.2f}",
                         help="Daily P&L from time decay")
                st.metric("📌 Rho", f"{greeks['rho']:.4f}",
                         help="1% rate change impact")
            
            st.subheader("📖 Greek Interpretations")
            
            interp_col1, interp_col2 = st.columns(2)
            
            with interp_col1:
                st.write("""
                **DELTA (Directional Exposure)**
                - Move ₹1 → Option changes ₹{:.4f}
                - 50% probability ITM if 0.50
                
                **GAMMA (Delta Acceleration)**
                - Delta changes by {:.6f} per ₹1
                - Higher gamma near ATM
                """.format(greeks['delta'], greeks['gamma']))
            
            with interp_col2:
                st.write("""
                **VEGA (Volatility Sensitivity)**
                - 1% vol increase → ₹{:.4f} change
                - Long vol = profit from vol rise
                
                **THETA (Time Decay)**
                - Lose ₹{:.2f} per day (if held)
                - Accelerates near expiry!
                """.format(greeks['vega'], greeks['theta']))
            
            with st.expander("🔬 Numerical Validation (Advanced)"):
                validation = model.validate_greeks_numerical()
                val_df = pd.DataFrame(validation).T
                st.dataframe(val_df, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error: {str(e)}")

with tab2:
    st.header("Interactive 3D Greeks Visualization")
    st.write("✨ **Hover over the surface to see exact values at each point!**")
    st.write("**Rotate, zoom, and pan the 3D surface to explore from all angles!**")
    
    viz_col1, viz_col2, viz_col3 = st.columns(3)
    
    with viz_col1:
        greek_choice = st.selectbox("Select Greek", ['delta', 'gamma', 'vega', 'theta'])
    with viz_col2:
        viz_type = st.selectbox("Visualization Type", ["Single Surface", "All 4 Surfaces", "Heatmap"])
    with viz_col3:
        st.write("")
        st.write("")
        generate_viz = st.button("🎨 Generate 3D Visualization", type="primary", use_container_width=True)
    
    if generate_viz:
        try:
            with st.spinner("Generating interactive 3D visualization..."):
                if viz_type == "Single Surface":
                    S_range = np.linspace(spot_price * 0.8, spot_price * 1.2, 25)
                    K_range = np.linspace(spot_price * 0.8, spot_price * 1.2, 25)
                    fig = IndianGreeksInteractiveVisualizer.plot_greeks_surface_interactive(
                        S_range, K_range, days_to_expiry, risk_free_rate, volatility, 'call', greek_choice
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                elif viz_type == "All 4 Surfaces":
                    fig = IndianGreeksInteractiveVisualizer.plot_all_greeks_interactive(
                        spot_price, days_to_expiry, risk_free_rate, volatility
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                elif viz_type == "Heatmap":
                    fig = IndianGreeksInteractiveVisualizer.plot_greeks_heatmap(
                        spot_price, days_to_expiry, risk_free_rate, volatility
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                st.success("✅ Interactive 3D visualization created! Hover to see values.")
        except Exception as e:
            st.error(f"Visualization error: {str(e)}")
    
    st.subheader("Other Interactive Visualizations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📊 Sensitivity Curves", use_container_width=True):
            try:
                with st.spinner("Generating sensitivity curves..."):
                    fig = IndianGreeksInteractiveVisualizer.plot_greek_sensitivity_interactive(
                        spot_price, spot_price, days_to_expiry, risk_free_rate, volatility, 'delta'
                    )
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    with col2:
        if st.button("📈 Volatility Impact", use_container_width=True):
            try:
                with st.spinner("Generating volatility impact..."):
                    fig = IndianGreeksInteractiveVisualizer.plot_volatility_impact_interactive(
                        spot_price, spot_price, days_to_expiry, risk_free_rate
                    )
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    col3, col4 = st.columns(2)
    
    with col3:
        if st.button("⏳ Time Decay Analysis", use_container_width=True):
            try:
                with st.spinner("Generating time decay..."):
                    fig = IndianGreeksInteractiveVisualizer.plot_time_decay_interactive(
                        spot_price, spot_price, risk_free_rate, volatility
                    )
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    with col4:
        if st.button("📉 IV Smile", use_container_width=True):
            try:
                with st.spinner("Generating volatility smile..."):
                    fig = IndianGreeksInteractiveVisualizer.plot_implied_volatility_smile_interactive(
                        spot_price, days_to_expiry, risk_free_rate, volatility
                    )
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error: {str(e)}")

with tab3:
    st.header("Multi-Leg Strategy Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Add Option Legs")
        
        leg_col1, leg_col2, leg_col3, leg_col4 = st.columns(4)
        
        with leg_col1:
            leg_strike = st.number_input("Strike", value=spot_price, key="leg_strike", step=10)
        with leg_col2:
            leg_type = st.selectbox("Type", ["Call", "Put"], key="leg_type")
        with leg_col3:
            leg_position = st.selectbox("Position", ["Long", "Short"], key="leg_position")
        with leg_col4:
            leg_qty = st.number_input("Qty", value=1, min_value=1, key="leg_qty")
        
        if st.button("➕ Add Leg", use_container_width=True):
            leg = IndianOptionLeg(
                S=spot_price,
                K=leg_strike,
                days_to_expiry=days_to_expiry,
                r=risk_free_rate,
                sigma=volatility,
                option_type=leg_type.lower(),
                position=leg_position.lower(),
                quantity=leg_qty
            )
            st.session_state.portfolio.add_leg(leg)
            st.success(f"Added: {leg_qty}x {leg_type} {leg_position} at ₹{leg_strike}")
    
    with col2:
        if st.button("🗑️ Clear Portfolio", use_container_width=True):
            st.session_state.portfolio.clear()
            st.info("Portfolio cleared")
    
    if st.session_state.portfolio.legs:
        st.subheader("Portfolio Details")
        greeks_df = st.session_state.portfolio.greeks_by_leg()
        st.dataframe(greeks_df, use_container_width=True, hide_index=True)
        
        st.subheader("💡 Risk Metrics")
        metrics = st.session_state.portfolio.risk_metrics()
        
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        
        with metric_col1:
            st.metric("Net Delta", f"{metrics['net_delta']:.4f}")
        with metric_col2:
            st.metric("Net Vega", f"{metrics['net_vega']:.4f}")
        with metric_col3:
            st.metric("Daily Theta", f"₹{metrics['daily_pnl_decay']:.2f}")
        with metric_col4:
            st.metric("Max P&L (1% move)", f"₹{metrics['max_profit_1pct']:.2f}")

with tab4:
    st.header("Pre-Built Strategies for Indian Market")
    
    strategy_choice = st.selectbox(
        "Select Strategy",
        [
            "Long Straddle (Expect Big Move)",
            "Short Straddle (Expect No Move)",
            "Bull Call Spread (Bullish, Limited Risk)",
            "Bear Put Spread (Bearish, Limited Risk)",
            "Iron Condor (Neutral, Premium Income)"
        ]
    )
    
    if st.button("Build Strategy", type="primary", use_container_width=True):
        st.session_state.portfolio.clear()
        
        if strategy_choice == "Long Straddle (Expect Big Move)":
            st.session_state.portfolio = IndianStrategyBuilder.long_straddle(
                spot_price, spot_price, days_to_expiry, risk_free_rate, volatility
            )
            st.info("**Long Straddle**: Buy Call + Put at same strike")
        
        elif strategy_choice == "Short Straddle (Expect No Move)":
            st.session_state.portfolio = IndianStrategyBuilder.short_straddle(
                spot_price, spot_price, days_to_expiry, risk_free_rate, volatility
            )
            st.warning("**Short Straddle**: Sell Call + Put (risky without stop loss!)")
        
        elif strategy_choice == "Bull Call Spread (Bullish, Limited Risk)":
            lower_strike = (spot_price // 100) * 100
            upper_strike = lower_strike + 200
            st.session_state.portfolio = IndianStrategyBuilder.bull_call_spread(
                spot_price, lower_strike, upper_strike, days_to_expiry, risk_free_rate, volatility
            )
            st.success("**Bull Call Spread**: Safe bullish strategy")
        
        elif strategy_choice == "Bear Put Spread (Bearish, Limited Risk)":
            upper_strike = (spot_price // 100) * 100
            lower_strike = upper_strike - 200
            st.session_state.portfolio = IndianStrategyBuilder.bear_put_spread(
                spot_price, upper_strike, lower_strike, days_to_expiry, risk_free_rate, volatility
            )
            st.success("**Bear Put Spread**: Safe bearish strategy")
        
        elif strategy_choice == "Iron Condor (Neutral, Premium Income)":
            strike_diff = IndianDataHelper.get_strike_difference(spot_price)
            put_short = spot_price - (4 * strike_diff)
            put_long = spot_price - (6 * strike_diff)
            call_short = spot_price + (4 * strike_diff)
            call_long = spot_price + (6 * strike_diff)
            
            st.session_state.portfolio = IndianStrategyBuilder.iron_condor(
                spot_price, put_short, put_long, call_short, call_long,
                days_to_expiry, risk_free_rate, volatility
            )
            st.success("**Iron Condor**: Premium income + limited risk")
        
        greeks_df = st.session_state.portfolio.greeks_by_leg()
        st.dataframe(greeks_df, use_container_width=True, hide_index=True)

with tab5:
    st.header("P&L & Risk Analysis")
    
    if st.session_state.portfolio.legs:
        col1, col2 = st.columns(2)
        
        with col1:
            move_percent = st.slider("% Move from Current Price", -20, 20, 10)
        
        with col2:
            vol_change = st.slider("Volatility Change", -0.20, 0.20, 0.0, 0.01)
        
        spot_min = spot_price * (1 - abs(move_percent) / 100)
        spot_max = spot_price * (1 + abs(move_percent) / 100)
        spot_range = np.linspace(spot_min, spot_max, 100)
        
        pnl, _ = st.session_state.portfolio.pnl_simulation(spot_range, vol_change)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.fill_between(spot_range, pnl, 0, where=(pnl > 0), color='green', alpha=0.3)
        ax.fill_between(spot_range, pnl, 0, where=(pnl <= 0), color='red', alpha=0.3)
        ax.plot(spot_range, pnl, color='blue', linewidth=2.5)
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax.axvline(x=spot_price, color='orange', linestyle='--', linewidth=1)
        ax.set_xlabel('Spot Price at Expiry (₹)')
        ax.set_ylabel('Profit / Loss (₹)')
        ax.set_title(f'Strategy P&L Analysis')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    else:
        st.info("Add positions in 'Multi-Leg Strategies' tab")

with tab6:
    st.header("🤖 Machine Learning - Volatility Regime Detection")
    
    st.write("Train an ML model to detect HIGH and LOW volatility regimes in Indian markets.")
    
    uploaded_file = st.file_uploader("Upload historical price data (CSV with 'date' and 'price' columns)", type=['csv'])
    
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("Data Preview:")
        st.dataframe(data.head(), use_container_width=True)
        
        if st.button("🤖 Train Regime Detection Model", type="primary", use_container_width=True):
            with st.spinner("Training ML model..."):
                try:
                    data.columns = ['date', 'price']
                    data['date'] = pd.to_datetime(data['date'])
                    data.set_index('date', inplace=True)
                    
                    prices = data['price']
                    returns = prices.pct_change().dropna()
                    
                    detector = IndianVolatilityRegimeDetector(model_type='random_forest')
                    features = detector.engineer_features(prices, returns)
                    labels = detector.create_regime_labels(returns)
                    
                    metrics = detector.train(features, labels)
                    
                    st.session_state.ml_detector = detector
                    
                    st.success("✅ Model trained successfully!")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Train Accuracy", f"{metrics['train_accuracy']:.2%}")
                    with col2:
                        st.metric("Test Accuracy", f"{metrics['test_accuracy']:.2%}")
                    with col3:
                        if 'roc_auc' in metrics:
                            st.metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")
                    
                    st.subheader("Feature Importance")
                    st.dataframe(metrics['feature_importance'].head(10), use_container_width=True)
                    
                    st.subheader("Classification Report")
                    st.text(metrics['classification_report'])
                
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    else:
        st.info("Upload historical price data to train the model")

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 0.9rem;'>
    <p>🇮🇳 Indian Options Greeks Calculator | Interactive 3D Visualization | ML Regime Detection</p>
    <p>Nifty • Bank Nifty • Sensex • Individual Stocks</p>
</div>
""", unsafe_allow_html=True)