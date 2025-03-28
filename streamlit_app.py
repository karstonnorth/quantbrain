import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pandas as pd
import json
import requests
from typing import List, Dict

# Configure the page
st.set_page_config(
    page_title="QuantBrain",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        margin-top: 1rem;
    }
    .css-1d391kg {
        padding-top: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'current_symbol' not in st.session_state:
    st.session_state.current_symbol = 'AAPL'
if 'sentiment_texts' not in st.session_state:
    st.session_state.sentiment_texts = []

# API endpoints
BASE_URL = "http://localhost:8000"

def fetch_financial_data(symbol: str, days: int = 30) -> pd.DataFrame:
    """Fetch financial data from the API"""
    endpoint = f"{BASE_URL}/api/data/financial"
    data = {
        "symbol": symbol,
        "start_date": (datetime.now() - timedelta(days=days)).isoformat(),
        "end_date": datetime.now().isoformat()
    }
    response = requests.post(endpoint, json=data)
    if response.status_code == 200:
        return pd.DataFrame(response.json())
    return pd.DataFrame()

def analyze_sentiment(texts: List[str]) -> List[Dict]:
    """Analyze sentiment using the API"""
    endpoint = f"{BASE_URL}/api/analysis/sentiment"
    data = {"texts": texts}
    response = requests.post(endpoint, json=data)
    if response.status_code == 200:
        return response.json()
    return []

def get_price_prediction(symbol: str, steps: int = 5) -> Dict:
    """Get price predictions from the API"""
    endpoint = f"{BASE_URL}/api/prediction/price"
    data = {"symbol": symbol, "steps": steps}
    response = requests.post(endpoint, json=data)
    if response.status_code == 200:
        return response.json()
    return {}

def get_company_info(symbol: str) -> Dict:
    """Get company information from the API"""
    endpoint = f"{BASE_URL}/api/company/{symbol}"
    response = requests.get(endpoint)
    if response.status_code == 200:
        return response.json()
    return {}

# Sidebar
st.sidebar.title("📊 QuantBrain")
st.sidebar.markdown("---")

# Navigation
page = st.sidebar.radio(
    "Navigation",
    ["Market Data", "Sentiment Analysis", "Price Prediction", "Company Info"]
)

# Main content
st.title("QuantBrain - AI-Powered Financial Analysis")

if page == "Market Data":
    st.header("Market Data Analysis")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        symbol = st.text_input("Stock Symbol", value=st.session_state.current_symbol)
        days = st.slider("Time Period (days)", 7, 365, 30)
    
    with col2:
        st.markdown("### Company Info")
        info = get_company_info(symbol)
        if info:
            st.write(f"**Name:** {info.get('name', 'N/A')}")
            st.write(f"**Sector:** {info.get('sector', 'N/A')}")
            st.write(f"**Industry:** {info.get('industry', 'N/A')}")
    
    if st.button("Fetch Data"):
        df = fetch_financial_data(symbol, days)
        if not df.empty:
            fig = go.Figure(data=[go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close']
            )])
            fig.update_layout(
                title=f"{symbol} Stock Price",
                yaxis_title="Price",
                xaxis_title="Date",
                template="plotly_dark"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Volume chart
            fig_volume = go.Figure(data=[go.Bar(
                x=df.index,
                y=df['Volume']
            )])
            fig_volume.update_layout(
                title="Trading Volume",
                yaxis_title="Volume",
                xaxis_title="Date",
                template="plotly_dark"
            )
            st.plotly_chart(fig_volume, use_container_width=True)

elif page == "Sentiment Analysis":
    st.header("Sentiment Analysis")
    
    st.markdown("### Enter Financial Text")
    new_text = st.text_area("Add new text for analysis")
    if st.button("Add Text"):
        if new_text:
            st.session_state.sentiment_texts.append(new_text)
            st.experimental_rerun()
    
    if st.session_state.sentiment_texts:
        st.markdown("### Analysis Results")
        results = analyze_sentiment(st.session_state.sentiment_texts)
        
        for text, result in zip(st.session_state.sentiment_texts, results):
            with st.expander(f"Text: {text[:50]}..."):
                st.write(f"**Sentiment:** {result['label']}")
                st.write(f"**Confidence:** {result['score']:.2%}")
        
        if st.button("Clear All"):
            st.session_state.sentiment_texts = []
            st.experimental_rerun()

elif page == "Price Prediction":
    st.header("Price Prediction")
    
    col1, col2 = st.columns(2)
    with col1:
        symbol = st.text_input("Stock Symbol", value=st.session_state.current_symbol)
    with col2:
        steps = st.slider("Prediction Steps", 1, 30, 5)
    
    if st.button("Generate Prediction"):
        prediction = get_price_prediction(symbol, steps)
        if prediction:
            st.markdown("### Price Predictions")
            
            # Create prediction dates
            last_date = datetime.strptime(prediction['last_date'], "%Y-%m-%d")
            dates = [last_date + timedelta(days=i+1) for i in range(steps)]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dates,
                y=prediction['predictions'],
                mode='lines+markers',
                name='Predicted Prices'
            ))
            
            fig.update_layout(
                title=f"{symbol} Price Predictions",
                yaxis_title="Price",
                xaxis_title="Date",
                template="plotly_dark"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display predictions in a table
            df_predictions = pd.DataFrame({
                'Date': dates,
                'Predicted Price': prediction['predictions']
            })
            st.dataframe(df_predictions)

elif page == "Company Info":
    st.header("Company Information")
    
    symbol = st.text_input("Stock Symbol", value=st.session_state.current_symbol)
    if st.button("Get Company Info"):
        info = get_company_info(symbol)
        if info:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### Basic Information")
                st.write(f"**Name:** {info.get('name', 'N/A')}")
                st.write(f"**Sector:** {info.get('sector', 'N/A')}")
                st.write(f"**Industry:** {info.get('industry', 'N/A')}")
            
            with col2:
                st.markdown("### Financial Metrics")
                st.write(f"**Market Cap:** ${info.get('market_cap', 0):,.2f}")
                st.write(f"**P/E Ratio:** {info.get('pe_ratio', 'N/A')}")
        else:
            st.error("Could not fetch company information")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit and FastAPI") 