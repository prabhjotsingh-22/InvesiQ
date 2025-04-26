import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from streamlit_option_menu import option_menu
from streamlit_extras.metric_cards import style_metric_cards
import yfinance as yf
from alpha_vantage.fundamentaldata import FundamentalData
from alpha_vantage.timeseries import TimeSeries
import os
from dotenv import load_dotenv
import google.generativeai as genai
from pdf_financial_analyzer import analyze_financial_report
import tempfile
import json
import time
import requests
from datetime import datetime, timedelta
from streamlit_lottie import st_lottie

# Load environment variables
load_dotenv()

# Configure API keys
ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

# Configure Gemini
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-2.0-flash')

# Initialize Alpha Vantage clients
fd = FundamentalData(key=ALPHA_VANTAGE_API_KEY)
ts = TimeSeries(key=ALPHA_VANTAGE_API_KEY, output_format='pandas')

# Page config
st.set_page_config(
    page_title="InvesiQ - Investment Intelligence, Redifined",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
    }
    .metric-card {
        background-color: #262730;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .css-1d391kg {
        padding: 1rem 1rem 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background-color: #262730;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1e1e1e;
        border-radius: 4px 4px 0px 0px;
        color: #ffffff;
        padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4a4a4a;
    }
    .stock-error {
        background-color: #ff4b4b20;
        border-left: 5px solid #ff4b4b;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0.5rem;
    }
    .stock-warning {
        background-color: #ffa50020;
        border-left: 5px solid #ffa500;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0.5rem;
    }
    .hero-title {
        font-size: 4rem;
        font-weight: 700;
        background: linear-gradient(45deg, #3494e6, #ec6ead);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .hero-subtitle {
        font-size: 1.5rem;
        color: #718096;
        margin-bottom: 2rem;
    }
    .feature-card {
        background-color: #1e1e1e;
        border-radius: 10px;
        padding: 20px;
        margin: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease;
    }
    .feature-card:hover {
        transform: translateY(-5px);
    }
    .feature-icon {
        font-size: 2rem;
        margin-bottom: 1rem;
    }
    .footer {
        position: fixed;
        bottom: 0;
        width: 100%;
        background-color: #1e1e1e;
        padding: 1rem;
        text-align: center;
        color: #718096;
    }
    .footer a {
        color: #3494e6;
        text-decoration: none;
    }
    .footer a:hover {
        text-decoration: underline;
    }
</style>
""", unsafe_allow_html=True)

def load_lottie_url(url: str):
    """Load Lottie animation from URL"""
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

def is_valid_stock_symbol(symbol):
    """Validate stock symbol format and existence"""
    if not symbol or not symbol.strip():
        return False
    
    # Basic format validation
    if not symbol.isalpha() or len(symbol) > 5:
        return False
    
    try:
        # Try to get basic info from Alpha Vantage
        url = f'https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={ALPHA_VANTAGE_API_KEY}'
        r = requests.get(url)
        data = r.json()
        
        if "Global Quote" in data and data["Global Quote"]:
            return True
        return False
    except:
        return False

def get_alpha_vantage_data(symbol):
    """Get stock data from Alpha Vantage as fallback"""
    try:
        data, meta = ts.get_daily(symbol=symbol, outputsize='full')
        data = data.head(252)  # Get approximately 1 year of trading days
        data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        return data
    except Exception as e:
        st.error(f"Error fetching Alpha Vantage data: {str(e)}")
        return None

@st.cache_data(ttl=3600)  # Cache data for 1 hour
def load_stock_data(symbol):
    """Load stock data using yfinance with fallback to Alpha Vantage"""
    if not is_valid_stock_symbol(symbol):
        st.error(f"Invalid stock symbol: {symbol}")
        return None, None
    
    try:
        # Try yfinance first
        stock = yf.Ticker(symbol)
        hist = stock.history(period="1y")
        
        if hist.empty:
            st.warning("No data available from Yahoo Finance. Trying Alpha Vantage...")
            hist = get_alpha_vantage_data(symbol)
            if hist is None or hist.empty:
                raise ValueError("No historical data available from either source")
            return None, hist  # Return only historical data
            
        # Try to get info with retries
        max_retries = 3
        for attempt in range(max_retries):
            try:
                info = stock.info
                return stock, hist
            except (json.JSONDecodeError, ValueError) as e:
                if attempt == max_retries - 1:
                    st.warning("Could not fetch detailed stock info. Showing limited information.")
                    return None, hist
                time.sleep(1)
                
    except Exception as e:
        st.error(f"Error loading stock data: {str(e)}")
        # Try Alpha Vantage as fallback
        hist = get_alpha_vantage_data(symbol)
        if hist is not None:
            st.info("Using Alpha Vantage data as fallback")
            return None, hist
        return None, None

@st.cache_data(ttl=3600)
def get_company_overview(symbol):
    """Get company overview from Alpha Vantage with caching"""
    try:
        overview, _ = fd.get_company_overview(symbol)
        return overview
    except Exception as e:
        st.warning(f"Could not fetch company overview: {str(e)}")
        return None

def analyze_metrics(overview):
    """Generate AI analysis of company metrics using Gemini"""
    try:
        prompt = f"""
        Analyze the following company metrics and provide key insights:
        {overview}
        
        Focus on:
        1. Financial Health
        2. Growth Potential
        3. Risk Factors
        4. Investment Considerations
        
        Provide a concise, bullet-point analysis.
        """
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating analysis: {str(e)}"

def create_stock_chart(hist):
    """Create stock price chart using plotly"""
    if hist is None or hist.empty:
        return None
        
    fig = go.Figure(data=[go.Candlestick(x=hist.index,
                open=hist['Open'],
                high=hist['High'],
                low=hist['Low'],
                close=hist['Close'])])
    
    fig.update_layout(
        title="Stock Price History",
        yaxis_title="Price",
        template="plotly_dark",
        height=500,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    return fig

# Sidebar navigation
with st.sidebar:
    selected = option_menu(
        menu_title="Navigation",
        options=["Home", "Finance Metrics Review", "Annual Report Analyzer"],
        icons=["house", "graph-up", "file-text"],
        menu_icon="cast",
        default_index=0,
    )

if selected == "Home":
    # Hero Section
    st.markdown('<h1 class="hero-title">InvesiQ</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">Investment Intelligence, Redefined.</p>', unsafe_allow_html=True)
    
    # Load and display animations
    col1, col2 = st.columns(2)
    
    with col1:
        lottie_investing = load_lottie_url("https://assets5.lottiefiles.com/packages/lf20_V9t630.json")
        if lottie_investing:
            st_lottie(lottie_investing, height=300, key="investing")
    
    with col2:
        lottie_analysis = load_lottie_url("https://assets5.lottiefiles.com/private_files/lf30_F3Bj3f.json")
        if lottie_analysis:
            st_lottie(lottie_analysis, height=300, key="analysis")
    
    # Features Section
    st.markdown("## 🚀 Features")
    
    feat_col1, feat_col2, feat_col3 = st.columns(3)
    
    with feat_col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">📈</div>
            <h3>Real-time Market Analysis</h3>
            <p>Get instant access to real-time stock data and market trends with advanced visualization tools.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with feat_col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🤖</div>
            <h3>AI-Powered Insights</h3>
            <p>Leverage cutting-edge AI technology to analyze financial reports and market data.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with feat_col3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">📊</div>
            <h3>Smart PDF Analysis</h3>
            <p>Extract and analyze key insights from annual reports and financial documents automatically.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # How It Works Section
    st.markdown("## 🔍 How It Works")
    st.markdown("""
    1. **Finance Metrics Review**: Enter any stock symbol to get real-time analysis and insights
    2. **Annual Report Analyzer**: Upload financial reports for AI-powered analysis
    3. **Get Actionable Insights**: Receive comprehensive analysis and investment recommendations
    """)

elif selected == "Finance Metrics Review":
    st.title("📈 Finance Metrics Review")
    
    # Stock symbol input with example and help
    col1, col2 = st.columns([2, 1])
    with col1:
        symbol = st.text_input(
            "Enter Stock Symbol",
            help="Enter a valid stock symbol (e.g., AAPL for Apple, MSFT for Microsoft)"
        ).upper()
    with col2:
        st.markdown("""
        **Example symbols:**
        - AAPL (Apple)
        - MSFT (Microsoft)
        - GOOGL (Google)
        """)
    
    if symbol:
        with st.spinner('Validating and loading stock data...'):
            stock, hist = load_stock_data(symbol)
            
            if hist is not None:  # We have at least historical data
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    fig = create_stock_chart(hist)
                    if fig is not None:
                        st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    if stock is not None and hasattr(stock, 'info'):
                        info = stock.info
                        st.subheader("Company Overview")
                        st.write(f"**{info.get('longName', symbol)}**")
                        st.write(f"Sector: {info.get('sector', 'N/A')}")
                        st.write(f"Industry: {info.get('industry', 'N/A')}")
                        
                        # Financial Metrics
                        st.subheader("Key Financial Metrics")
                        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                        
                        with metrics_col1:
                            st.metric("Market Cap", f"${info.get('marketCap', 0):,.0f}")
                        with metrics_col2:
                            st.metric("P/E Ratio", f"{info.get('trailingPE', 0):.2f}")
                        with metrics_col3:
                            st.metric("Dividend Yield", f"{info.get('dividendYield', 0):.2%}")
                    else:
                        st.info("Limited data available. Some metrics may not be shown.")
                
                # Get detailed company overview
                overview = get_company_overview(symbol)
                if overview is not None:
                    st.subheader("AI Analysis")
                    with st.spinner('Generating AI analysis...'):
                        analysis = analyze_metrics(overview)
                        st.markdown(analysis)
                else:
                    st.warning("Could not fetch detailed company analysis.")

elif selected == "Annual Report Analyzer":
    st.title("📑 Annual Report Analyzer")
    
    uploaded_file = st.file_uploader("Upload Annual Report PDF", type="pdf")
    
    if uploaded_file is not None:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        with st.spinner('Analyzing the report... This may take a few minutes.'):
            try:
                # Analyze the PDF
                analysis = analyze_financial_report(tmp_path)
                
                # Display analysis in sections
                st.subheader("Analysis Results")
                
                # Create tabs for different sections
                tab1, tab2, tab3, tab4 = st.tabs([
                    "Executive Summary",
                    "Financial Health",
                    "Risks & Opportunities",
                    "Investment Recommendation"
                ])
                
                # Split analysis into sections (assuming the analysis has clear section markers)
                sections = analysis.split('\n\n')
                
                with tab1:
                    st.markdown(sections[0] if len(sections) > 0 else "No executive summary available")
                with tab2:
                    st.markdown(sections[1] if len(sections) > 1 else "No financial health analysis available")
                with tab3:
                    st.markdown(sections[2] if len(sections) > 2 else "No risks and opportunities analysis available")
                with tab4:
                    st.markdown(sections[3] if len(sections) > 3 else "No investment recommendation available")
                
            except Exception as e:
                st.error(f"Error analyzing PDF: {str(e)}")
            finally:
                # Clean up the temporary file
                os.unlink(tmp_path)
    
    # Tips section
    with st.expander("Tips for Best Results"):
        st.markdown("""
        - Upload a clear, searchable PDF (not scanned images)
        - Ensure the report is in English
        - Annual reports (10-K) and quarterly reports (10-Q) work best
        - Large files may take longer to analyze
        """)

# Footer
st.markdown("""
<div class="footer">
    Made with ❤️ by MongoDBoys (
    <a href="https://github.com/dbestvarun" target="_blank">Varun</a> and 
    <a href="https://github.com/prabhjotsingh-22" target="_blank">Prabhjot</a>)
</div>
""", unsafe_allow_html=True) 