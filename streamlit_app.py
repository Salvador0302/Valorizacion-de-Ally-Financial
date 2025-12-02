"""
Ally Financial Valuation Dashboard
===================================
Interactive Streamlit dashboard for stock valuation analysis.

Run with: streamlit run streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, '.')

from src.data_loader import DataLoader
from src.valuation import ValuationEngine
from src.monte_carlo import MonteCarloSimulation

# Page configuration
st.set_page_config(
    page_title="ALLY Valuation Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .recommendation-buy {
        color: #28a745;
        font-weight: bold;
    }
    .recommendation-sell {
        color: #dc3545;
        font-weight: bold;
    }
    .recommendation-hold {
        color: #ffc107;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">📊 Ally Financial (ALLY) Valuation Dashboard</h1>', unsafe_allow_html=True)

# Sidebar
st.sidebar.header("⚙️ Configuration")

# Ticker selection (for extensibility)
ticker = st.sidebar.text_input("Stock Ticker", value="ALLY", max_chars=5)

# Valuation parameters
st.sidebar.subheader("📊 Valuation Parameters")
required_return = st.sidebar.slider("Required Return (WACC)", 0.05, 0.20, 0.10, 0.01, format="%.2f")
growth_rate = st.sidebar.slider("Growth Rate (Stage 1)", 0.01, 0.15, 0.05, 0.01, format="%.2f")
terminal_growth = st.sidebar.slider("Terminal Growth Rate", 0.01, 0.05, 0.02, 0.005, format="%.3f")
forecast_years = st.sidebar.slider("Forecast Years", 3, 10, 5)

# Monte Carlo parameters
st.sidebar.subheader("🎲 Monte Carlo Settings")
n_simulations = st.sidebar.selectbox("Number of Simulations", [1000, 5000, 10000, 50000], index=2)
mc_days = st.sidebar.selectbox("Forecast Days", [63, 126, 252, 504], index=2, 
                                format_func=lambda x: f"{x} days ({x//252} year{'s' if x//252 > 1 else ''})" if x >= 252 else f"{x} days ({x//21} months)")

# Initialize data with caching
@st.cache_data(ttl=3600)
def load_data(ticker_symbol):
    """Load and cache financial data."""
    loader = DataLoader(ticker=ticker_symbol)
    prices = loader.get_historical_prices(period="5y")
    summary = loader.get_summary()
    return loader, prices, summary

@st.cache_data(ttl=3600)
def run_valuations(ticker_symbol, req_return, growth, term_growth, years):
    """Run and cache valuation calculations."""
    loader = DataLoader(ticker=ticker_symbol)
    valuation = ValuationEngine(data_loader=loader)
    
    book_value = valuation.book_value_valuation()
    adj_book = valuation.adjusted_book_value_valuation()
    pe_val = valuation.pe_ratio_valuation()
    ddm = valuation.dividend_discount_model(
        required_return=req_return,
        growth_rate=growth,
        terminal_growth=term_growth,
        forecast_years=years
    )
    comparables = valuation.comparable_companies_valuation()
    dcf = valuation.dcf_fcf_valuation(
        required_return=req_return,
        growth_rate=growth,
        terminal_growth=term_growth,
        forecast_years=years
    )
    
    summary_df = valuation.get_valuation_summary()
    fair_value = valuation.get_fair_value_estimate()
    
    return {
        'book_value': book_value,
        'adj_book': adj_book,
        'pe_val': pe_val,
        'ddm': ddm,
        'comparables': comparables,
        'dcf': dcf,
        'summary': summary_df,
        'fair_value': fair_value
    }

@st.cache_data(ttl=3600)
def run_monte_carlo(prices_data, n_sims, days):
    """Run and cache Monte Carlo simulation."""
    mc = MonteCarloSimulation(n_simulations=n_sims, n_days=days, random_seed=42)
    return mc.run_simulation(prices_data)

# Load data
try:
    with st.spinner("Loading financial data..."):
        loader, prices, summary = load_data(ticker)
    
    # Main content
    # Row 1: Key Metrics
    st.header("📈 Key Financial Metrics")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    current_price = summary.get('current_price', 0)
    
    with col1:
        st.metric("Current Price", f"${current_price:.2f}")
    with col2:
        market_cap = summary.get('market_cap', 0)
        st.metric("Market Cap", f"${market_cap/1e9:.2f}B")
    with col3:
        eps = summary.get('eps', 0)
        st.metric("EPS", f"${eps:.2f}")
    with col4:
        div_yield = summary.get('dividend_yield', 0) * 100
        st.metric("Dividend Yield", f"{div_yield:.2f}%")
    with col5:
        book_val = summary.get('book_value_per_share', 0)
        st.metric("Book Value/Share", f"${book_val:.2f}")
    
    # Row 2: Price Chart
    st.header("📊 Historical Price Data")
    
    # Create interactive price chart
    fig_price = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True,
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3],
        subplot_titles=('Price', 'Volume')
    )
    
    fig_price.add_trace(
        go.Candlestick(
            x=prices.index,
            open=prices['Open'],
            high=prices['High'],
            low=prices['Low'],
            close=prices['Close'],
            name='OHLC'
        ),
        row=1, col=1
    )
    
    fig_price.add_trace(
        go.Bar(x=prices.index, y=prices['Volume'], name='Volume', marker_color='rgba(31, 119, 180, 0.5)'),
        row=2, col=1
    )
    
    fig_price.update_layout(
        height=600,
        showlegend=False,
        xaxis_rangeslider_visible=False,
        title_text=f"{ticker} - 5 Year Price History"
    )
    
    st.plotly_chart(fig_price, use_container_width=True)
    
    # Run valuations
    st.header("💰 Valuation Analysis")
    
    with st.spinner("Running valuation models..."):
        valuations = run_valuations(ticker, required_return, growth_rate, terminal_growth, forecast_years)
    
    # Create tabs for different valuation methods
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📗 Book Value", "📘 Adjusted Book", "📈 P/E Ratio", 
        "💵 DDM", "🏢 Comparables", "💎 DCF/FCF"
    ])
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Book Value Analysis")
            bv = valuations['book_value']
            st.metric("Intrinsic Value", f"${bv['intrinsic_value']:.2f}")
            st.metric("Price to Book", f"{bv['price_to_book']:.2f}x")
            st.metric("Upside Potential", f"{bv['upside_potential']:.2f}%", 
                     delta=f"{bv['upside_potential']:.1f}%")
        with col2:
            st.write("**Method Description:**")
            st.write("Book Value = Total Equity / Shares Outstanding")
            st.write(f"- Total Equity: ${bv.get('total_equity', 0):,.0f}")
            st.write(f"- Shares Outstanding: {bv.get('shares_outstanding', 0):,.0f}")
    
    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Adjusted Book Value Analysis")
            abv = valuations['adj_book']
            st.metric("Intrinsic Value", f"${abv['intrinsic_value']:.2f}")
            st.metric("Price to Adj. Book", f"{abv['price_to_adjusted_book']:.2f}x")
            st.metric("Upside Potential", f"{abv['upside_potential']:.2f}%",
                     delta=f"{abv['upside_potential']:.1f}%")
        with col2:
            st.write("**Method Description:**")
            st.write("Adjusted Book = (Total Equity - Intangibles) / Shares")
            st.write(f"- Total Equity: ${abv.get('total_equity', 0):,.0f}")
            st.write(f"- Intangible Assets: ${abv.get('intangible_assets', 0):,.0f}")
            st.write(f"- Adjusted Equity: ${abv.get('adjusted_equity', 0):,.0f}")
    
    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("P/E Ratio Analysis")
            pe = valuations['pe_val']
            st.metric("Intrinsic Value", f"${pe['intrinsic_value']:.2f}")
            st.metric("Current P/E", f"{pe['current_pe']:.2f}x")
            st.metric("Target P/E (Industry)", f"{pe['target_pe']:.2f}x")
            st.metric("Upside Potential", f"{pe['upside_potential']:.2f}%",
                     delta=f"{pe['upside_potential']:.1f}%")
        with col2:
            st.write("**Method Description:**")
            st.write("Intrinsic Value = EPS × Target P/E")
            st.write(f"- EPS: ${pe.get('eps', 0):.2f}")
            st.write(f"- Target P/E: {pe.get('target_pe', 0):.2f}x (Industry Median)")
    
    with tab4:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Dividend Discount Model")
            ddm = valuations['ddm']
            if ddm.get('error'):
                st.warning(ddm['error'])
            else:
                st.metric("Intrinsic Value", f"${ddm['intrinsic_value']:.2f}")
                st.metric("Dividend/Share", f"${ddm['dividend_per_share']:.2f}")
                st.metric("Upside Potential", f"{ddm['upside_potential']:.2f}%",
                         delta=f"{ddm['upside_potential']:.1f}%")
        with col2:
            st.write("**Model Parameters:**")
            st.write(f"- Required Return: {ddm.get('required_return', 0)*100:.1f}%")
            st.write(f"- Growth Rate (Stage 1): {ddm.get('growth_rate_stage1', 0)*100:.1f}%")
            st.write(f"- Terminal Growth: {ddm.get('terminal_growth_rate', 0)*100:.1f}%")
            st.write(f"- Forecast Years: {ddm.get('forecast_years', 0)}")
            
            if ddm.get('projected_dividends'):
                st.write("**Projected Dividends:**")
                div_df = pd.DataFrame(ddm['projected_dividends'])
                st.dataframe(div_df.round(2))
    
    with tab5:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Comparable Companies")
            comp = valuations['comparables']
            if comp.get('error'):
                st.warning(comp['error'])
            else:
                st.metric("Intrinsic Value", f"${comp['intrinsic_value']:.2f}")
                st.metric("Implied Value (P/E)", f"${comp.get('implied_value_pe', 0):.2f}")
                st.metric("Implied Value (P/B)", f"${comp.get('implied_value_pb', 0):.2f}")
                st.metric("Upside Potential", f"{comp['upside_potential']:.2f}%",
                         delta=f"{comp['upside_potential']:.1f}%")
        with col2:
            st.write("**Peer Company Metrics:**")
            if comp.get('peer_metrics'):
                peer_df = pd.DataFrame(comp['peer_metrics'])
                st.dataframe(peer_df)
                st.write(f"Median P/E: {comp.get('median_pe', 'N/A')}")
                st.write(f"Median P/B: {comp.get('median_pb', 'N/A')}")
    
    with tab6:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("DCF (Free Cash Flow)")
            dcf = valuations['dcf']
            if dcf.get('error'):
                st.warning(dcf['error'])
            else:
                st.metric("Intrinsic Value", f"${dcf['intrinsic_value']:.2f}")
                st.metric("Free Cash Flow", f"${dcf['free_cash_flow']:,.0f}")
                st.metric("Enterprise Value", f"${dcf['enterprise_value']:,.0f}")
                st.metric("Upside Potential", f"{dcf['upside_potential']:.2f}%",
                         delta=f"{dcf['upside_potential']:.1f}%")
        with col2:
            st.write("**Model Parameters:**")
            st.write(f"- WACC: {dcf.get('wacc', 0)*100:.1f}%")
            st.write(f"- Growth Rate: {dcf.get('growth_rate', 0)*100:.1f}%")
            st.write(f"- Terminal Growth: {dcf.get('terminal_growth_rate', 0)*100:.1f}%")
            
            if dcf.get('projected_fcfs'):
                st.write("**Projected Free Cash Flows:**")
                fcf_df = pd.DataFrame(dcf['projected_fcfs'])
                fcf_df['fcf'] = fcf_df['fcf'].apply(lambda x: f"${x:,.0f}")
                fcf_df['pv'] = fcf_df['pv'].apply(lambda x: f"${x:,.0f}")
                st.dataframe(fcf_df)
    
    # Valuation Comparison Chart
    st.header("📊 Valuation Methods Comparison")
    
    summary_df = valuations['summary']
    valid_vals = summary_df[summary_df['Intrinsic Value ($)'] > 0].copy()
    
    # Create comparison bar chart
    fig_comparison = go.Figure()
    
    colors = ['#28a745' if v > current_price else '#dc3545' for v in valid_vals['Intrinsic Value ($)']]
    
    fig_comparison.add_trace(go.Bar(
        x=valid_vals['Method'],
        y=valid_vals['Intrinsic Value ($)'],
        marker_color=colors,
        text=[f"${v:.2f}" for v in valid_vals['Intrinsic Value ($)']],
        textposition='outside',
        name='Intrinsic Value'
    ))
    
    fig_comparison.add_hline(
        y=current_price, 
        line_dash="dash", 
        line_color="blue",
        annotation_text=f"Current Price: ${current_price:.2f}",
        annotation_position="top right"
    )
    
    fair_val = valuations['fair_value']['fair_value_estimate']
    fig_comparison.add_hline(
        y=fair_val, 
        line_dash="dot", 
        line_color="purple",
        annotation_text=f"Fair Value Est.: ${fair_val:.2f}",
        annotation_position="bottom right"
    )
    
    fig_comparison.update_layout(
        title="Intrinsic Value by Valuation Method",
        xaxis_title="Valuation Method",
        yaxis_title="Price ($)",
        height=500
    )
    
    st.plotly_chart(fig_comparison, use_container_width=True)
    
    # Fair Value Summary
    col1, col2, col3 = st.columns(3)
    
    fair_value_data = valuations['fair_value']
    upside = fair_value_data['upside_potential']
    
    with col1:
        st.metric("📊 Weighted Fair Value", f"${fair_value_data['fair_value_estimate']:.2f}")
    with col2:
        st.metric("💵 Current Price", f"${fair_value_data['current_price']:.2f}")
    with col3:
        delta_color = "normal" if upside > 0 else "inverse"
        st.metric("📈 Upside Potential", f"{upside:.2f}%", delta=f"{upside:.1f}%", delta_color=delta_color)
    
    # Monte Carlo Simulation
    st.header("🎲 Monte Carlo Simulation")
    
    with st.spinner("Running Monte Carlo simulation..."):
        close_prices = prices['Close']
        mc_results = run_monte_carlo(close_prices, n_simulations, mc_days)
    
    # Monte Carlo Results
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Mean Final Price", f"${mc_results['mean_final_price']:.2f}")
    with col2:
        st.metric("Probability of Profit", f"{mc_results['prob_profit']:.1f}%")
    with col3:
        st.metric("95% VaR", f"${mc_results['var_95']:.2f}")
    with col4:
        st.metric("Expected Return", f"{mc_results['expected_return_pct']:.2f}%")
    
    # Monte Carlo Charts
    mc_col1, mc_col2 = st.columns(2)
    
    with mc_col1:
        # Price paths
        fig_paths = go.Figure()
        
        # Plot sample paths
        n_sample_paths = 100
        for i in range(min(n_sample_paths, mc_results['price_paths'].shape[0])):
            fig_paths.add_trace(go.Scatter(
                y=mc_results['price_paths'][i],
                mode='lines',
                line=dict(width=0.5, color='rgba(31, 119, 180, 0.1)'),
                showlegend=False
            ))
        
        # Add mean path
        mean_path = np.mean(mc_results['price_paths'], axis=0)
        fig_paths.add_trace(go.Scatter(
            y=mean_path,
            mode='lines',
            line=dict(width=2, color='red'),
            name='Mean Path'
        ))
        
        fig_paths.update_layout(
            title=f"Monte Carlo Price Paths ({n_simulations:,} simulations)",
            xaxis_title="Trading Days",
            yaxis_title="Price ($)",
            height=400
        )
        
        st.plotly_chart(fig_paths, use_container_width=True)
    
    with mc_col2:
        # Final price distribution
        fig_dist = go.Figure()
        
        fig_dist.add_trace(go.Histogram(
            x=mc_results['final_prices'],
            nbinsx=50,
            name='Final Prices',
            marker_color='rgba(31, 119, 180, 0.7)'
        ))
        
        fig_dist.add_vline(x=mc_results['initial_price'], line_dash="dash", line_color="green",
                          annotation_text=f"Initial: ${mc_results['initial_price']:.2f}")
        fig_dist.add_vline(x=mc_results['mean_final_price'], line_dash="solid", line_color="red",
                          annotation_text=f"Mean: ${mc_results['mean_final_price']:.2f}")
        
        fig_dist.update_layout(
            title="Distribution of Final Prices",
            xaxis_title="Price ($)",
            yaxis_title="Frequency",
            height=400
        )
        
        st.plotly_chart(fig_dist, use_container_width=True)
    
    # Monte Carlo Statistics Table
    st.subheader("📊 Monte Carlo Statistics")
    
    mc_stats = {
        "Metric": [
            "Initial Price", "Mean Final Price", "Median Final Price", "Std Dev",
            "5th Percentile", "25th Percentile", "75th Percentile", "95th Percentile",
            "VaR (95%)", "CVaR (95%)", "Annual Volatility"
        ],
        "Value": [
            f"${mc_results['initial_price']:.2f}",
            f"${mc_results['mean_final_price']:.2f}",
            f"${mc_results['median_final_price']:.2f}",
            f"${mc_results['std_final_price']:.2f}",
            f"${mc_results['percentiles'][5]:.2f}",
            f"${mc_results['percentiles'][25]:.2f}",
            f"${mc_results['percentiles'][75]:.2f}",
            f"${mc_results['percentiles'][95]:.2f}",
            f"${mc_results['var_95']:.2f}",
            f"${mc_results['cvar_95']:.2f}",
            f"{mc_results['volatility_annual']*100:.2f}%"
        ]
    }
    
    st.dataframe(pd.DataFrame(mc_stats), use_container_width=True)
    
    # LSTM Section (Optional)
    st.header("🤖 LSTM Price Prediction (Optional)")
    
    run_lstm = st.checkbox("Run LSTM Model (may take a few minutes)")
    
    if run_lstm:
        try:
            from src.lstm_model import LSTMPredictor
            
            with st.spinner("Training LSTM model... This may take a few minutes."):
                lstm = LSTMPredictor(
                    sequence_length=60,
                    units=50,
                    epochs=25,
                    batch_size=32
                )
                
                training_results = lstm.train(close_prices, train_ratio=0.8, verbose=0)
                
                # Predict future
                days_ahead = 30
                future_predictions = lstm.predict_future(close_prices, days_ahead=days_ahead)
                
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Training RMSE", f"${training_results['train_rmse']:.2f}")
            with col2:
                st.metric("Test RMSE", f"${training_results['test_rmse']:.2f}")
            with col3:
                predicted_change = ((future_predictions[-1] - close_prices.iloc[-1]) / close_prices.iloc[-1] * 100)
                st.metric(f"Predicted Price ({days_ahead}d)", f"${future_predictions[-1]:.2f}",
                         delta=f"{predicted_change:.2f}%")
            
            # LSTM Prediction Chart
            fig_lstm = go.Figure()
            
            # Historical data (last 90 days)
            hist_data = close_prices.iloc[-90:]
            fig_lstm.add_trace(go.Scatter(
                x=hist_data.index,
                y=hist_data.values,
                mode='lines',
                name='Historical',
                line=dict(color='blue', width=2)
            ))
            
            # Future predictions
            future_dates = pd.date_range(start=close_prices.index[-1] + pd.Timedelta(days=1), 
                                        periods=days_ahead, freq='B')
            fig_lstm.add_trace(go.Scatter(
                x=future_dates,
                y=future_predictions,
                mode='lines',
                name='LSTM Forecast',
                line=dict(color='red', width=2, dash='dash')
            ))
            
            fig_lstm.update_layout(
                title=f"LSTM {days_ahead}-Day Price Forecast",
                xaxis_title="Date",
                yaxis_title="Price ($)",
                height=400
            )
            
            st.plotly_chart(fig_lstm, use_container_width=True)
            
        except ImportError:
            st.warning("TensorFlow is required for LSTM predictions. Install with: pip install tensorflow")
        except Exception as e:
            st.error(f"Error running LSTM model: {str(e)}")
    
    # Final Recommendation
    st.header("📋 Investment Recommendation")
    
    upside = fair_value_data['upside_potential']
    
    if upside > 20:
        recommendation = "STRONG BUY"
        rec_color = "recommendation-buy"
        description = "The stock appears significantly undervalued based on multiple valuation methods."
    elif upside > 10:
        recommendation = "BUY"
        rec_color = "recommendation-buy"
        description = "The stock appears moderately undervalued and may offer attractive returns."
    elif upside > -10:
        recommendation = "HOLD"
        rec_color = "recommendation-hold"
        description = "The stock appears fairly valued at current levels."
    elif upside > -20:
        recommendation = "SELL"
        rec_color = "recommendation-sell"
        description = "The stock appears moderately overvalued at current prices."
    else:
        recommendation = "STRONG SELL"
        rec_color = "recommendation-sell"
        description = "The stock appears significantly overvalued based on multiple valuation methods."
    
    st.markdown(f"""
    <div class="metric-card">
        <h2 class="{rec_color}">{recommendation}</h2>
        <p>{description}</p>
        <ul>
            <li>Current Price: ${current_price:.2f}</li>
            <li>Fair Value Estimate: ${fair_value_data['fair_value_estimate']:.2f}</li>
            <li>Upside/Downside Potential: {upside:.2f}%</li>
            <li>Monte Carlo Probability of Profit (1Y): {mc_results['prob_profit']:.1f}%</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.warning("""
    ⚠️ **Disclaimer**: This analysis is for educational purposes only and should not be considered 
    as financial advice. Always conduct your own research and consult with a qualified financial 
    advisor before making investment decisions. Past performance does not guarantee future results.
    """)

except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    st.info("Please check your internet connection and ensure the ticker symbol is valid.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>Built with Streamlit | Data from Yahoo Finance | © 2024</p>
</div>
""", unsafe_allow_html=True)
