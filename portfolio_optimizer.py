import matplotlib.pyplot as plt
import seaborn as sns
# Manually register seaborn styles if not available
try:
    if "seaborn-deep" not in plt.style.available:
        sns.set_theme()  # This will register seaborn styles with matplotlib
except Exception as e:
    print(f"Could not register seaborn styles: {e}")

import streamlit as st
from pypfopt import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt import plotting
import copy
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go  # Add this import for candlestick charts
from plotly.subplots import make_subplots
from datetime import datetime
from io import BytesIO # To display figures for efficient frontier later
# from statsmodels.tsa.arima.model import ARIMA
import os
# import yfinance as yf  # Replaced with yahooquery
from yahooquery import Ticker
import json
import requests
from bs4 import BeautifulSoup
import re
import matplotlib.dates as mdates
from openai import OpenAI
import pickle
import time  # Added for managing delays between API calls
# Add these imports at the top of your file 
from datetime import datetime  # Add this if not present
from scipy import stats        # Add this new import
import warnings               # Add this new import
warnings.filterwarnings('ignore')  # Add this new line

# Initialize session state variables if they don't exist
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'current_ticker' not in st.session_state:
    st.session_state.current_ticker = None
if 'ticker_data' not in st.session_state:
    st.session_state.ticker_data = {}
if 'portfolio_tickers' not in st.session_state:
    st.session_state.portfolio_tickers = []
# Store portfolio optimization results
if 'portfolio_weights' not in st.session_state:
    st.session_state.portfolio_weights = {}
if 'portfolio_performance' not in st.session_state:
    st.session_state.portfolio_performance = {}
if 'portfolio_sharpe' not in st.session_state:
    st.session_state.portfolio_sharpe = None
if 'portfolio_volatility' not in st.session_state:
    st.session_state.portfolio_volatility = None
if 'portfolio_expected_return' not in st.session_state:
    st.session_state.portfolio_expected_return = None

# Function to get OpenAI API Key
def get_openai_api_key():
    api_key = st.sidebar.text_input("Enter OpenAI API Key:", type="password")
    if not api_key:
        st.sidebar.warning("Please enter your OpenAI API key to use the chatbot features.")
    return api_key

# Initialize OpenAI client
def get_openai_client(api_key):
    if api_key:
        try:
            return OpenAI(api_key=api_key)
        except TypeError:
            # Fall back to older version initialization if needed
            import openai
            openai.api_key = api_key
            return openai
    return None

# Function to extract stock ticker from user query using OpenAI's API
def extract_ticker(client, query):
    # Check for common non-ticker query patterns
    general_query_patterns = [
        'high growth', 'low risk', 'dividend', 'growth stocks', 'blue chip',
        'penny stocks', 'investment strategy', 'etf', 'index fund', 'mutual fund',
        'sector', 'industry', 'market', 'economy', 'what stocks', 'which stocks',
        'recommend', 'suggestion', 'portfolio', 'diversification'
    ]
    
    # First check if the query is likely a general question rather than about a specific ticker
    query_lower = query.lower()
    for pattern in general_query_patterns:
        if pattern in query_lower:
            log_error(f"Detected general query pattern: {pattern} in: {query}")
            return None, None
    
    # If not a general query, use OpenAI to extract ticker
    tools = [
        {
            "type": "function",
            "function": {
                "name": "extract_stock_ticker",
                "description": "Extract stock ticker symbols and company names from user queries",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "ticker": {
                            "type": "string",
                            "description": "The stock ticker symbol mentioned in the query (e.g., AAPL for Apple). Return null if no specific ticker is mentioned."
                        },
                        "company_name": {
                            "type": "string",
                            "description": "The company name if mentioned (e.g., Apple). Return null if no specific company is mentioned."
                        }
                    },
                    "required": ["ticker"]
                }
            }
        }
    ]
    
    try:
        # Use OpenAI's chat completion to extract ticker
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a financial assistant that extracts stock ticker symbols from queries. Only extract actual ticker symbols. If the query is asking for recommendations, suggestions, or general categories of stocks without mentioning a specific ticker, return null."},
                {"role": "user", "content": query}
            ],
            tools=tools,
            tool_choice={"type": "function", "function": {"name": "extract_stock_ticker"}}
        )
        
        # New format - process function calling response
        if hasattr(response, 'choices') and len(response.choices) > 0:
            message = response.choices[0].message
            if hasattr(message, 'tool_calls') and message.tool_calls:
                tool_call = message.tool_calls[0]
                function_args = json.loads(tool_call.function.arguments)
                ticker = function_args.get("ticker")
                company_name = function_args.get("company_name")
                
                # Further validation to avoid treating general terms as tickers
                if ticker and ticker.upper() in ["HIGH GROWTH", "LOW RISK", "DIVIDEND", "GROWTH", "TECH", "BLUE CHIP"]:
                    log_error(f"Prevented general term being treated as ticker: {ticker}")
                    return None, None
                    
                return ticker, company_name
        
        # If we reached here, there was an issue with extracting the ticker
        log_error(f"Failed to extract ticker from: {query}")
        return None, None
        
    except Exception as e:
        st.error(f"Error extracting ticker: {e}")
        log_error(f"Error in extract_ticker: {e}")
        return None, None

# Function to generate dummy stock data for testing
def generate_dummy_stock_data(ticker, start_date, end_date):
    """Generate simulated stock data for a ticker when real data isn't available"""
    try:
        # Convert string dates to datetime objects if they aren't already
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
            
        # Create a date range
        date_range = pd.date_range(start=start_date, end=end_date, freq='B')
        
        # Generate random price data based on ticker (to get consistent data for the same ticker)
        np.random.seed(hash(ticker) % 2**32)
        
        # Start with a price between $10 and $100
        initial_price = np.random.uniform(10, 100)
        
        # Generate daily returns with slight upward bias
        daily_returns = np.random.normal(0.0005, 0.02, size=len(date_range))
        
        # Calculate price series
        price_series = initial_price * (1 + np.cumsum(daily_returns))
        
        # Create a DataFrame with OHLC data
        df = pd.DataFrame(index=date_range)
        df['Close'] = price_series
        df['Open'] = df['Close'].shift(1) * (1 + np.random.normal(0, 0.003, size=len(df)))
        df['High'] = np.maximum(df['Open'], df['Close']) * (1 + np.abs(np.random.normal(0, 0.005, size=len(df))))
        df['Low'] = np.minimum(df['Open'], df['Close']) * (1 - np.abs(np.random.normal(0, 0.005, size=len(df))))
        df['Volume'] = np.random.randint(100000, 5000000, size=len(df))
        
        # Fill NaN values
        df = df.fillna(method='bfill')
        
        # For consistency, convert index to date objects
        df.index = [idx.date() for idx in df.index]
        
        # Log that we used dummy data
        log_error(f"Generated dummy data for {ticker} from {start_date} to {end_date}")
        
        return df
    except Exception as e:
        log_error(f"Error generating dummy data for {ticker}: {e}")
        # Return a minimal dataframe for extreme fallback
        df = pd.DataFrame(index=[datetime.today().date()], data={'Close': [50.0]})
        return df

# Function to fetch stock data
def fetch_stock_data(ticker, period="1y"):
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # Create Ticker object
            stock = Ticker(ticker)
            
            # Get historical data
            history = stock.history(period=period)
            
            # Handle multi-index and reformat to match the expected structure
            if isinstance(history.index, pd.MultiIndex):
                # This selects just the data for the specific ticker
                history = history.xs(ticker, level=0, drop_level=True)
            
            # Clean index
            history = history[~history.index.duplicated(keep='first')]
            
            # Ensure index is datetime
            history.index = pd.to_datetime(history.index)
            
            # Convert datetime index to date objects for consistency
            history.index = [idx.date() for idx in history.index]
            
            # Get stock info
            info = stock.asset_profile
            # If the info is a dictionary of dictionaries (for multiple tickers), extract just this ticker's info
            if isinstance(info, dict) and ticker in info:
                info = info[ticker]
            
            # Add additional info from quote_type and summary_detail
            quote_info = stock.quote_type
            if isinstance(quote_info, dict) and ticker in quote_info:
                quote_info = quote_info[ticker]
                
            price_info = stock.summary_detail
            if isinstance(price_info, dict) and ticker in price_info:
                price_info = price_info[ticker]
            
            # Combine all info into a single dict similar to yf.Ticker.info
            combined_info = {}
            for source in [info, quote_info, price_info]:
                if isinstance(source, dict):
                    combined_info.update(source)
            
            log_error(f"Successfully fetched data for {ticker}")
            
            return {
                "history": history,
                "info": combined_info
            }
        except Exception as e:
            if attempt < max_retries - 1:
                st.warning(f"Retry {attempt+1}/{max_retries} for {ticker} after error: {e}")
                time.sleep(2)  # Wait between retries
            else:
                st.error(f"Error fetching data for {ticker} after {max_retries} attempts: {e}")
                log_error(f"Failed to get ticker '{ticker}' reason: {e}")
                return None

# Function to get stock price chart
def get_stock_price_chart(ticker, price_history):
    try:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=price_history.index,
            y=price_history['close'],  # lowercase
            mode='lines',
            name='Close Price',
            line=dict(color='blue', width=2)
        ))
        
        fig.update_layout(
            title=f'{ticker} Stock Price',
            xaxis_title='Date',
            yaxis_title='Price ($)',
            height=400,
            showlegend=True
        )
        
        return fig
        
    except Exception as e:
        log_error(f"Error in get_stock_price_chart for {ticker}: {e}")
        return go.Figure()

# Function to get candlestick chart
def get_candlestick_chart(ticker, price_history):
    try:
        fig = go.Figure(data=go.Candlestick(
            x=price_history.index,
            open=price_history['open'],    # lowercase
            high=price_history['high'],    # lowercase
            low=price_history['low'],      # lowercase
            close=price_history['close'],  # lowercase
            name=ticker
        ))
        
        fig.update_layout(
            title=f'{ticker} Candlestick Chart',
            xaxis_title='Date',
            yaxis_title='Price ($)',
            height=400,
            xaxis_rangeslider_visible=False
        )
        
        return fig
        
    except Exception as e:
        log_error(f"Error in get_candlestick_chart for {ticker}: {e}")
        return go.Figure()

# Function to get stock financial metrics
def get_financial_metrics(stock_data):
    """Debug version to see what financial data we actually have - FIXED"""
    
    # Show debug info in an expander so it doesn't clutter the UI
    with st.expander("🔍 Debug Info (click to expand)", expanded=False):
        st.write("**stock_data structure:**")
        st.write(f"Keys available: {list(stock_data.keys())}")
        
        # Check the 'info' data specifically
        if "info" in stock_data:
            info = stock_data["info"]
            st.write(f"info type: {type(info)}")
            st.write(f"info empty?: {not bool(info)}")
            
            if info and isinstance(info, dict):
                st.write(f"info has {len(info)} keys")
                st.write(f"First 10 keys: {list(info.keys())[:10]}")
                
                # Check specific financial keys
                financial_keys = [
                    "marketCap", "trailingPE", "trailingEPS", "dividendYield",
                    "fiftyTwoWeekHigh", "fiftyTwoWeekLow", "averageVolume",
                    "forwardPE", "bookValue", "priceToBook", "beta"
                ]
                
                st.write("**Financial key availability:**")
                for key in financial_keys:
                    value = info.get(key)
                    status = "✅" if value is not None else "❌"
                    st.write(f"{status} {key}: {value}")
                
            else:
                st.write("❌ info is empty or not a dictionary")
        else:
            st.write("❌ No 'info' key found in stock_data")
        
        # Check history data too
        if "history" in stock_data:
            history = stock_data["history"]
            if history is not None and not history.empty:
                st.write(f"✅ History data: {history.shape[0]} rows, {history.shape[1]} columns")
                st.write(f"Columns: {list(history.columns)}")
            else:
                st.write("❌ History data is empty")
    
    # Now try to extract metrics normally
    info = stock_data.get("info", {})
    
    # If info is empty, return basic message
    if not info:
        st.warning("⚠️ No fundamental data available from API. This is common with free data sources.")
        return {
            "Market Cap": "N/A (No API data)",
            "P/E Ratio": "N/A (No API data)",
            "EPS": "N/A (No API data)",
            "Dividend Yield": "N/A (No API data)",
            "52 Week High": "N/A (No API data)",
            "52 Week Low": "N/A (No API data)",
            "Average Volume": "N/A (No API data)"
        }
    
    # Extract metrics with the data we have
    metrics = {}
    
    # Market Cap
    market_cap = info.get("marketCap")
    if market_cap and isinstance(market_cap, (int, float)) and market_cap > 0:
        if market_cap >= 1e9:
            metrics["Market Cap"] = f"${market_cap/1e9:.2f}B"
        else:
            metrics["Market Cap"] = f"${market_cap/1e6:.2f}M"
    else:
        metrics["Market Cap"] = "N/A"
    
    # P/E Ratio
    pe = info.get("trailingPE") or info.get("forwardPE")
    metrics["P/E Ratio"] = f"{pe:.2f}" if pe and pe > 0 else "N/A"
    
    # EPS
    eps = info.get("trailingEPS")
    metrics["EPS"] = f"${eps:.2f}" if eps else "N/A"
    
    # Dividend Yield
    div_yield = info.get("dividendYield")
    metrics["Dividend Yield"] = f"{div_yield*100:.2f}%" if div_yield else "N/A"
    
    # 52 Week High/Low
    high_52 = info.get("fiftyTwoWeekHigh")
    low_52 = info.get("fiftyTwoWeekLow")
    metrics["52 Week High"] = f"${high_52:.2f}" if high_52 else "N/A"
    metrics["52 Week Low"] = f"${low_52:.2f}" if low_52 else "N/A"
    
    # Average Volume
    volume = info.get("averageVolume")
    if volume and volume > 0:
        metrics["Average Volume"] = f"{volume/1e6:.1f}M" if volume >= 1e6 else f"{volume:,.0f}"
    else:
        metrics["Average Volume"] = "N/A"
    
    return metrics

# Function to scrape recent news about a ticker
def get_recent_news(ticker):
    try:
        url = f"https://finance.yahoo.com/quote/{ticker}/news"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        news_items = []
        for item in soup.select('h3'):
            if item.text and len(item.text.strip()) > 10:
                news_items.append(item.text.strip())
                if len(news_items) >= 5:  # Get top 5 news items
                    break
                    
        return news_items
    except Exception as e:
        return [f"Error fetching news: {e}"]

# Function to analyze stock data using LLM
def analyze_stock(client, ticker, stock_data, news):
    # Prepare context for the LLM
    info = stock_data["info"]
    history = stock_data["history"]
    
    # Calculate some additional metrics
    current_price = history['Close'].iloc[-1] if not history.empty else "N/A"
    price_change_1d = (history['Close'].iloc[-1] - history['Close'].iloc[-2]) / history['Close'].iloc[-2] * 100 if len(history) >= 2 else "N/A"
    price_change_1w = (history['Close'].iloc[-1] - history['Close'].iloc[-5]) / history['Close'].iloc[-5] * 100 if len(history) >= 5 else "N/A"
    price_change_1m = (history['Close'].iloc[-1] - history['Close'].iloc[-20]) / history['Close'].iloc[-20] * 100 if len(history) >= 20 else "N/A"
    
    # Create context
    context = f"""
    Analyzing {ticker} ({info.get('shortName', ticker)}):
    
    Current Price: ${current_price if current_price != 'N/A' else 'N/A'}
    Price Change (1 day): {price_change_1d if price_change_1d != 'N/A' else 'N/A'}%
    Price Change (1 week): {price_change_1w if price_change_1w != 'N/A' else 'N/A'}%
    Price Change (1 month): {price_change_1m if price_change_1m != 'N/A' else 'N/A'}%
    
    Key Metrics:
    - Market Cap: ${info.get('marketCap', 'N/A')}
    - P/E Ratio: {info.get('trailingPE', 'N/A')}
    - EPS: {info.get('trailingEPS', 'N/A')}
    - Dividend Yield: {info.get('dividendYield', 'N/A') if info.get('dividendYield') else 'N/A'}
    - 52 Week Range: ${info.get('fiftyTwoWeekLow', 'N/A')} - ${info.get('fiftyTwoWeekHigh', 'N/A')}
    
    Recent News Headlines:
    """
    
    for i, item in enumerate(news, 1):
        context += f"\n{i}. {item}"
    
    try:
        # Get analysis from OpenAI - new client format
        if hasattr(client, 'chat') and hasattr(client.chat, 'completions'):
            response = client.chat.completions.create(
                model="gpt-3.5-turbo-0125",
                messages=[
                    {"role": "system", "content": "You are a financial analyst providing insights on stocks. Give a balanced view considering recent performance, news, and metrics. Include potential risks and opportunities. Format your response with clear sections and bullet points. Keep your analysis concise, insightful, and actionable."},
                    {"role": "user", "content": context}
                ]
            )
            return response.choices[0].message.content
        # Legacy OpenAI format
        else:
            response = client.ChatCompletion.create(
                model="gpt-3.5-turbo-0125",
                messages=[
                    {"role": "system", "content": "You are a financial analyst providing insights on stocks. Give a balanced view considering recent performance, news, and metrics. Include potential risks and opportunities. Format your response with clear sections and bullet points. Keep your analysis concise, insightful, and actionable."},
                    {"role": "user", "content": context}
                ]
            )
            return response['choices'][0]['message']['content']
    except Exception as e:
        st.error(f"Error analyzing stock: {e}")
        return f"Sorry, I encountered an error analyzing {ticker}: {str(e)}"

# Function to calculate technical indicators
def calculate_technical_indicators(history):
    df = history.copy()
    
    # Make sure we have a Close column, or use close if available (yahooquery uses lowercase)
    if 'close' in df.columns and 'Close' not in df.columns:
        df['Close'] = df['close']
    
    # Calculate moving averages
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    
    # Calculate Relative Strength Index (RSI)
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD (Moving Average Convergence Divergence)
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    return df

# Function to plot technical indicators
def plot_technical_indicators(df, ticker):
    # Create a figure with 2 subplots (price with MAs and RSI)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1, 1]})
    
    # Plot price and moving averages
    ax1.plot(df.index, df['Close'], label='Close Price')
    ax1.plot(df.index, df['SMA_20'], label='SMA 20')
    ax1.plot(df.index, df['SMA_50'], label='SMA 50')
    ax1.set_title(f'{ticker} Price and Technical Indicators')
    ax1.set_ylabel('Price ($)')
    ax1.legend()
    ax1.grid(True)
    
    # Plot MACD
    ax2.plot(df.index, df['MACD'], label='MACD')
    ax2.plot(df.index, df['Signal_Line'], label='Signal Line')
    ax2.bar(df.index, df['MACD'] - df['Signal_Line'], alpha=0.5, label='Histogram')
    ax2.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    ax2.set_ylabel('MACD')
    ax2.legend()
    ax2.grid(True)
    
    # Plot RSI
    ax3.plot(df.index, df['RSI'], label='RSI')
    ax3.axhline(y=70, color='r', linestyle='--', alpha=0.5)
    ax3.axhline(y=30, color='g', linestyle='--', alpha=0.5)
    ax3.set_ylabel('RSI')
    ax3.set_xlabel('Date')
    ax3.legend()
    ax3.grid(True)
    
    # Format x-axis dates
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return fig

# Function to process user query
def process_query(client, query):
    try:
        # Extract ticker from user query
        ticker, company_name = extract_ticker(client, query)
        
        # Check if query is about investment strategy with portfolio stocks
        investment_strategy_terms = ["investment strategy", "invest", "allocate", "allocation", "portfolio strategy", 
                                     "risk tolerance", "diversification", "diversify", "$", "dollar", "money"]
        is_investment_query = any(term in query.lower() for term in investment_strategy_terms)
        
        # Check if query is about portfolio stocks generally
        portfolio_related_terms = ["portfolio", "selected stocks", "my stocks", "these stocks", "the stocks", "i have selected"]
        is_portfolio_query = any(term in query.lower() for term in portfolio_related_terms)
        
        # If query seems to be about investment strategy and we have portfolio optimization data
        if is_investment_query and is_portfolio_query and st.session_state.portfolio_tickers:
            portfolio_tickers_str = ", ".join(st.session_state.portfolio_tickers)
            
            # Format portfolio metrics
            portfolio_metrics = ""
            if st.session_state.portfolio_sharpe is not None:
                portfolio_metrics += f"\nPortfolio Optimization Results:"
                portfolio_metrics += f"\n- Sharpe Ratio: {st.session_state.portfolio_sharpe:.2f}"
                portfolio_metrics += f"\n- Expected Annual Return: {st.session_state.portfolio_expected_return*100:.2f}%"
                portfolio_metrics += f"\n- Annual Volatility: {st.session_state.portfolio_volatility*100:.2f}%"
            
            # Format portfolio weights if available
            weights_info = ""
            if st.session_state.portfolio_weights:
                weights_info = "\nOptimized Portfolio Weights:"
                for ticker, weight in st.session_state.portfolio_weights.items():
                    weights_info += f"\n- {ticker}: {weight*100:.2f}%"
            
            prompt = f"""
            The user has selected the following stocks in their portfolio: {portfolio_tickers_str}.
            
            Their query about investment strategy is: "{query}"
            
            {portfolio_metrics}
            {weights_info}
            
            Provide a detailed investment strategy based on:
            1. The portfolio optimization results shown above (if available)
            2. The specific stocks in their portfolio
            3. Their risk tolerance mentioned in the query
            4. The investment amount mentioned in the query (if any)
            
            Include specific allocation recommendations and explain the rationale behind them.
            Consider the optimization metrics like Sharpe ratio, expected return and volatility in your recommendations.
            """
            
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000
            )
            
            analysis = response.choices[0].message.content
            return analysis
        
        # If just about portfolio generally (not investment strategy specific)
        elif is_portfolio_query and st.session_state.portfolio_tickers:
            portfolio_tickers_str = ", ".join(st.session_state.portfolio_tickers)
            prompt = f"""
            The user has selected the following stocks in their portfolio: {portfolio_tickers_str}.
            
            Their query is: "{query}"
            
            Provide a detailed analysis about these specific stocks in their portfolio.
            Include a brief overview of each stock's recent performance, potential outlook, 
            and how they might work together in a portfolio.
            """
            
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=800
            )
            
            analysis = response.choices[0].message.content
            return analysis
        
        if ticker is None or ticker.upper() == 'NONE':
            # No valid ticker found, handle as a general query
            return general_query_response(client, query)
        
        # Fetch stock data using the extracted ticker symbol
        stock_data = fetch_stock_data(ticker)
        
        if stock_data is None or stock_data.get("history") is None or stock_data["history"].empty:
            # Failed to get data for the ticker
            st.error(f"Could not retrieve data for {ticker}. Please try a different query.")
            return general_query_response(client, query)
        
        # Analyze and display stock information
        analyze_stock(ticker, stock_data)
        
        # Generate a response using OpenAI based on the stock data
        prompt = f"""
        Provide a brief analysis for {ticker} ({company_name if company_name else ticker}) based on recent price action and fundamentals if available.
        If there's a specific aspect of the query: "{query}", focus on that. 
        Keep the response under 400 words and make sure it's complete and not cut off.
        """
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800
        )
        
        analysis = response.choices[0].message.content
        return analysis
        
    except Exception as e:
        st.error(f"Error processing query: {e}")
        log_error(f"Error in process_query: {e}")
        return "I encountered an error processing your request. Please try a different query."

# Function to handle general queries (when no ticker is found)
def general_query_response(client, query):
    """Handle general financial queries when no specific ticker is found."""
    
    # Check if the query is about app features
    app_feature_keywords = ["chart", "graph", "visualization", "candlestick", "line chart", 
                           "feature", "can you", "do you", "available", "function", 
                           "optimize", "portfolio", "what can", "how to use", "help"]
    
    is_app_question = any(keyword in query.lower() for keyword in app_feature_keywords)
    
    # Add context about the user's selected portfolio stocks if they exist
    portfolio_context = ""
    if st.session_state.portfolio_tickers:
        portfolio_tickers_str = ", ".join(st.session_state.portfolio_tickers)
        portfolio_context = f"The user has selected the following stocks in their portfolio: {portfolio_tickers_str}."
    
    if is_app_question:
        system_content = f"""You are a helpful assistant explaining the features of a financial app.
        This app has the following capabilities:
        
        1. Portfolio Optimization:
           - Allows users to select stocks and optimize their portfolio weights using modern portfolio theory
           - Shows efficient frontier visualization
           - Calculates expected returns, volatility, and Sharpe ratio
        
        2. Stock Visualization and Analysis:
           - Line charts showing historical stock prices
           - Candlestick charts for detailed price movement analysis
           - Technical indicators including Moving Averages, RSI, and MACD
           - Financial metrics display for each stock
        
        3. Stock News:
           - Fetches and displays recent news articles about selected stocks
        
        4. AI-powered Assistance:
           - Answers general questions about investing and finance
           - Provides analysis of specific stocks when queried
        
        {portfolio_context}
        
        When asked about features, explain the relevant functionality in a helpful way.
        Provide complete answers and make sure your responses are not cut off.
        """
    else:
        system_content = f"""You are a financial advisor specializing in stocks and investments. 
        Provide helpful, educational responses about investing, stocks, and financial markets. 
        If asked about specific stocks but can't identify a ticker, politely ask for clarification.
        
        {portfolio_context}
        
        If the user's query appears to be related to their portfolio stocks, include relevant information
        about those specific stocks in your response.
        
        Provide complete answers and make sure your responses are not cut off.
        """
    
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": query}
    ]
    
    try:
        # Use the OpenAI API to get a response
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=1200
        )
        return response.choices[0].message.content
    except Exception as e:
        log_error(f"Error in general_query_response: {e}")
        return f"I'm sorry, but I encountered an error processing your general query: {str(e)}"

# Function to plot cumulative returns
def plot_cum_returns(data, title):    
    daily_cum_returns = 1 + data.dropna().pct_change()
    daily_cum_returns = daily_cum_returns.cumprod()*100
    fig = px.line(daily_cum_returns, title=title)
    return fig
    
# Function to plot efficient frontier and max Sharpe ratio
def plot_efficient_frontier_and_max_sharpe(mu, S): 
    """Optimize portfolio for max Sharpe ratio and plot efficient frontier with solver fallback"""
    try:
        # Create the plot
        fig, ax = plt.subplots(figsize=(6,4))
        
        # Try multiple solvers for the efficient frontier
        ef = EfficientFrontier(mu, S)
        ef_max_sharpe = copy.deepcopy(ef)
        
        # Try plotting efficient frontier with different approaches
        try:
            plotting.plot_efficient_frontier(ef, ax=ax, show_assets=False)
        except Exception as e:
            try:
                # If plotting fails, create a simple manual efficient frontier
                returns_range = np.linspace(mu.min(), mu.max(), 50)
                risk_range = []
                
                for target_return in returns_range:
                    try:
                        ef_temp = EfficientFrontier(mu, S)
                        ef_temp.efficient_return(target_return)
                        _, vol, _ = ef_temp.portfolio_performance()
                        risk_range.append(vol)
                    except:
                        risk_range.append(np.nan)
                
                # Plot the frontier
                valid_mask = ~np.isnan(risk_range)
                if np.any(valid_mask):
                    ax.plot(np.array(risk_range)[valid_mask], 
                           returns_range[valid_mask], 'b-', label='Efficient Frontier')
                else:
                    ax.text(0.5, 0.7, 'Efficient Frontier\nCalculation Failed', 
                           ha='center', va='center', transform=ax.transAxes)
            except:
                ax.text(0.5, 0.7, 'Efficient Frontier\nCalculation Failed', 
                       ha='center', va='center', transform=ax.transAxes)

        # Find the max sharpe portfolio (solver is set at EfficientFrontier level, not max_sharpe level)
        try:
            ef_max_sharpe.max_sharpe(risk_free_rate=0.02)
            ret_tangent, std_tangent, _ = ef_max_sharpe.portfolio_performance()
            ax.scatter(std_tangent, ret_tangent, marker="*", s=100, c="r", label="Max Sharpe")
        except Exception as e:
            try:
                # Try with regularized covariance matrix if original fails
                S_reg = S + np.eye(len(S)) * 1e-6
                ef_max_sharpe_reg = EfficientFrontier(mu, S_reg)
                ef_max_sharpe_reg.max_sharpe(risk_free_rate=0.02)
                ret_tangent, std_tangent, _ = ef_max_sharpe_reg.portfolio_performance()
                ax.scatter(std_tangent, ret_tangent, marker="*", s=100, c="r", label="Max Sharpe")
            except Exception as e2:
                st.warning(f"Could not calculate max Sharpe portfolio for plotting: {e2}")

        # Generate random portfolios for visualization
        try:
            n_samples = 1000
            w = np.random.dirichlet(np.ones(ef.n_assets), n_samples)
            rets = w.dot(ef.expected_returns)
            stds = np.sqrt(np.diag(w @ ef.cov_matrix @ w.T)) 
            sharpes = rets / stds
            ax.scatter(stds, rets, marker=".", c=sharpes, cmap="viridis_r", alpha=0.6)
        except Exception as e:
            st.warning(f"Could not generate random portfolios for visualization: {e}")

        # Finalize plot
        ax.set_xlabel('Risk (Standard Deviation)')
        ax.set_ylabel('Expected Return')
        ax.set_title('Efficient Frontier')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
        
    except Exception as e:
        # Create a fallback plot if everything fails
        fig, ax = plt.subplots(figsize=(6,4))
        ax.text(0.5, 0.5, f'Efficient Frontier\nVisualization Failed\n{str(e)}', 
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Efficient Frontier (Error)')
        return fig

# Function to cache stock data
def cache_stock_data(ticker, data):
    cache_dir = "cache"
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{ticker}.pkl")
    with open(cache_file, "wb") as f:
        pickle.dump(data, f)

# Function to load cached stock data
def load_cached_stock_data(ticker):
    cache_file = os.path.join("cache", f"{ticker}.pkl")
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            return pickle.load(f)
    return None

# Enhanced error logging function
def log_error(message):
    with open("error_log.txt", "a") as log_file:
        log_file.write(f"{datetime.now()}: {message}\n")

# Function to plot moving averages
def plot_moving_averages(ticker, price_history):
    try:
        # Create a copy of the dataframe
        df = price_history.copy()
        
        # Calculate moving averages using lowercase 'close'
        df['MA20'] = df['close'].rolling(window=20).mean()
        df['MA50'] = df['close'].rolling(window=50).mean()
        df['MA200'] = df['close'].rolling(window=200).mean()
        
        # Create the plot
        fig = go.Figure()
        
        # Add price line using lowercase 'close'
        fig.add_trace(go.Scatter(x=df.index, y=df['close'], mode='lines', name='Price', line=dict(color='blue')))
        
        # Add MA lines
        fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], mode='lines', name='20-day MA', line=dict(color='orange')))
        fig.add_trace(go.Scatter(x=df.index, y=df['MA50'], mode='lines', name='50-day MA', line=dict(color='green')))
        fig.add_trace(go.Scatter(x=df.index, y=df['MA200'], mode='lines', name='200-day MA', line=dict(color='red')))
        
        # Update layout
        fig.update_layout(title=f'{ticker} - Moving Averages', xaxis_title='Date', yaxis_title='Price (USD)',
                         legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        
        return fig
    except Exception as e:
        log_error(f"Error in plot_moving_averages for {ticker}: {e}")
        return go.Figure()


# Function to calculate and plot RSI
def plot_rsi(ticker, price_history, window=14):
    try:
        # Create a copy of the dataframe
        df = price_history.copy()
        
        # Calculate RSI using lowercase 'close'
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Create the plot
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                          vertical_spacing=0.1, 
                          row_heights=[0.7, 0.3])
        
        # Add price to the first subplot using lowercase 'close'
        fig.add_trace(go.Scatter(x=df.index, y=df['close'], mode='lines', name='Price', line=dict(color='blue')), row=1, col=1)
        
        # Add RSI to the second subplot
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], mode='lines', name='RSI', line=dict(color='purple')), row=2, col=1)
        
        # Add overbought/oversold lines
        fig.add_trace(go.Scatter(x=df.index, y=[70]*len(df), mode='lines', name='Overbought', 
                               line=dict(color='red', dash='dash')), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=[30]*len(df), mode='lines', name='Oversold', 
                               line=dict(color='green', dash='dash')), row=2, col=1)
        
        # Update layout
        fig.update_layout(title=f'{ticker} - RSI (14)', xaxis_title='Date', height=600,
                         legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        fig.update_yaxes(title_text='Price (USD)', row=1, col=1)
        fig.update_yaxes(title_text='RSI', row=2, col=1)
        
        return fig
    except Exception as e:
        log_error(f"Error in plot_rsi for {ticker}: {e}")
        return go.Figure()

# Function to calculate and plot MACD
def plot_macd(ticker, price_history):
    try:
        # Create a copy of the dataframe
        df = price_history.copy()
        
        # Calculate MACD using lowercase 'close'
        close_prices = df['close']
        exp1 = close_prices.ewm(span=12).mean()
        exp2 = close_prices.ewm(span=26).mean()
        df['MACD'] = exp1 - exp2
        df['Signal'] = df['MACD'].ewm(span=9).mean()
        df['Histogram'] = df['MACD'] - df['Signal']
        
        # Create the plot
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                          vertical_spacing=0.1, 
                          row_heights=[0.5, 0.3, 0.2])
        
        # Add price to the first subplot using lowercase 'close'
        fig.add_trace(go.Scatter(x=df.index, y=df['close'], mode='lines', name='Price', line=dict(color='blue')), row=1, col=1)
        
        # Add MACD and Signal to the second subplot
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], mode='lines', name='MACD', line=dict(color='red')), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Signal'], mode='lines', name='Signal', line=dict(color='orange')), row=2, col=1)
        
        # Add Histogram to the third subplot
        colors = ['red' if val < 0 else 'green' for val in df['Histogram']]
        fig.add_trace(go.Bar(x=df.index, y=df['Histogram'], name='Histogram', marker_color=colors), row=3, col=1)
        
        # Update layout
        fig.update_layout(title=f'{ticker} - MACD', xaxis_title='Date', height=800,
                         legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        fig.update_yaxes(title_text='Price (USD)', row=1, col=1)
        fig.update_yaxes(title_text='MACD', row=2, col=1)
        fig.update_yaxes(title_text='Histogram', row=3, col=1)
        
        return fig
    except Exception as e:
        log_error(f"Error in plot_macd for {ticker}: {e}")
        return go.Figure()

# Function to normalize yahooquery data
def normalize_yahoo_data(df):
    """
    Normalize yahooquery data to ensure consistent column names and index format
    """
    # Make a copy to avoid modifying the original dataframe
    df = df.copy()
    
    # Handle column name differences (yahooquery uses lowercase)
    column_map = {
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume',
        'adjclose': 'Adj Close'
    }
    
    # Normalize column names
    for lower, upper in column_map.items():
        if lower in df.columns and upper not in df.columns:
            df.rename(columns={lower: upper}, inplace=True)
    
    # Ensure index is datetime
    if len(df) > 0:
        df.index = pd.to_datetime(df.index)
        # Convert datetime index to date objects for consistency
        df.index = [idx.date() for idx in df.index]
    
    return df

# Update the fetch_stock_data function to use the normalizer
def fetch_stock_data(ticker, period="1y"):
    """Fetch stock data from Yahoo Finance using yahooquery - TIMEZONE FIXED VERSION"""
    try:
        # Create a Ticker object for this symbol
        stock = Ticker(ticker)
        
        # Fetch stock data
        history = stock.history(period=period)
        
        # Handle multi-index result (yahooquery returns MultiIndex with ticker as first level)
        if isinstance(history.index, pd.MultiIndex):
            # This selects just the data for the specific ticker
            history = history.xs(ticker, level=0, drop_level=True)
        
        if history.empty:
            return None
        
        # TIMEZONE FIX - Remove timezone information to avoid mixing tz-aware with tz-naive
        if hasattr(history.index, 'tz') and history.index.tz is not None:
            history.index = history.index.tz_localize(None)
        
        # Ensure index is datetime (without timezone)
        history.index = pd.to_datetime(history.index)
        
        # Clean and process the data
        history = history[~history.index.duplicated(keep='first')]
        
        # Return the processed data
        return {
            "history": history,
            "info": {
                "longName": f"{ticker} Stock",
                "sector": "Technology",  # Default for your tech focus
                "regularMarketPrice": history['Close'].iloc[-1] if 'Close' in history.columns else None
            }
        }
        
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None
# Function to analyze stock data
def analyze_stock(ticker, stock_data):
    if not stock_data:
        st.error(f"No data available for {ticker}")
        return
    
    try:
        price_history = stock_data["history"]
        info = stock_data["info"]
        
        # Check if we have the necessary data
        if price_history.empty:
            st.error(f"No price history available for {ticker}")
            return
        
        # Get company name from info, with ticker as fallback
        company_name = info.get("longName", info.get("shortName", ticker))
        
        # Display the company name and ticker symbol
        st.subheader(f"{company_name} ({ticker})")
        
        # Show company business summary if available
        if "longBusinessSummary" in info:
            with st.expander("Company Description"):
                st.write(info["longBusinessSummary"])
        elif "description" in info:
            with st.expander("Company Description"):
                st.write(info["description"])
        
        # Display stock price chart
        st.write("## Stock Price Chart")
        chart_tabs = st.tabs(["Line Chart", "Candlestick Chart"])
        
        with chart_tabs[0]:
            fig = get_stock_price_chart(ticker, price_history)
            st.plotly_chart(fig, use_container_width=True)
        
        with chart_tabs[1]:
            fig = get_candlestick_chart(ticker, price_history)
            st.plotly_chart(fig, use_container_width=True)
        
        # Calculate and display key financial metrics
        st.write("## Key Financial Metrics")
        metrics = get_financial_metrics(stock_data)
        
        # Format metrics into three columns
        col1, col2, col3 = st.columns(3)
        metrics_list = list(metrics.items())
        
        # Split metrics across columns
        for i, (key, value) in enumerate(metrics_list):
            with [col1, col2, col3][i % 3]:
                if key == "Market Cap" and isinstance(value, (int, float)):
                    value = f"${value/1e9:.2f}B" if value >= 1e9 else f"${value/1e6:.2f}M"
                elif key == "Dividend Yield" and isinstance(value, (int, float)):
                    value = f"{value*100:.2f}%" if value else 'N/A'
                st.metric(key, value)
        
        # Recent News
        st.write("## Recent News")
        news = get_recent_news(ticker)
        if news:
            for article in news[:5]:  # Display up to 5 news articles
                st.markdown(f"### [{article}]({article})")
                st.write("---")
        else:
            st.write("No recent news available")
        
        # Technical Analysis
        st.write("## Technical Analysis")
        tech_tabs = st.tabs(["Moving Averages", "RSI", "MACD"])
        
        with tech_tabs[0]:
            try:
                fig = plot_moving_averages(ticker, price_history)
                if fig and hasattr(fig, 'data') and len(fig.data) > 0:
                    st.plotly_chart(fig, use_container_width=True, key=f"ma_chart_{ticker}")
                else:
                    st.warning("Unable to generate moving averages chart")
            except Exception as e:
                st.error(f"Moving averages error: {e}")
        
        with tech_tabs[1]:
            try:
                fig = plot_rsi(ticker, price_history)
                if fig and hasattr(fig, 'data') and len(fig.data) > 0:
                    st.plotly_chart(fig, use_container_width=True, key=f"rsi_chart_{ticker}")
                else:
                    st.warning("Unable to generate RSI chart")
            except Exception as e:
                st.error(f"RSI error: {e}")
        
        with tech_tabs[2]:
            try:
                fig = plot_macd(ticker, price_history)
                if fig and hasattr(fig, 'data') and len(fig.data) > 0:
                    st.plotly_chart(fig, use_container_width=True, key=f"macd_chart_{ticker}")
                else:
                    st.warning("Unable to generate MACD chart")
            except Exception as e:
                st.error(f"MACD error: {e}")
                
    except Exception as e:
        st.error(f"Error analyzing {ticker}: {e}")
        log_error(f"Error analyzing {ticker}: {e}")

def load_portfolio_data(tickers, start_date, end_date):
    """Function to load stock data for the portfolio optimizer"""
    stocks_df = pd.DataFrame()
    successful_tickers = []
    use_dummy_data = True
    
    if not tickers:
        st.warning("Please select at least 2 tickers to build a portfolio.")
        return None, []
    
    # Fetch stock prices for each selected ticker
    for ticker in tickers:
        try:
            # Add a delay between requests to avoid rate limiting
            time.sleep(2)
            
            # Try multiple times with error handling
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    # Create a Ticker object for this symbol
                    stock = Ticker(ticker)
                    
                    # Fetch stock data with proper date parameters
                    stock_data = stock.history(
                        start=start_date,
                        end=end_date,
                    )
                    
                    # Handle multi-index result (yahooquery returns MultiIndex with ticker as first level)
                    if isinstance(stock_data.index, pd.MultiIndex):
                        # This selects just the data for the specific ticker
                        stock_data = stock_data.xs(ticker, level=0, drop_level=True)
                    
                    if stock_data.empty:
                        if attempt < max_retries - 1:
                            time.sleep(3)
                            continue
                        elif use_dummy_data:
                            # Generate dummy data if API fails after all retries
                            stock_data = generate_dummy_stock_data(ticker, start_date, end_date)
                            st.warning(f"Using dummy data for {ticker} after failed API calls")
                        else:
                            error_message = f"Could not retrieve data for {ticker} after {max_retries} attempts"
                            st.error(error_message)
                            break
                    
                    # TIMEZONE FIX - Remove timezone information to make everything tz-naive
                    if hasattr(stock_data.index, 'tz') and stock_data.index.tz is not None:
                        stock_data.index = stock_data.index.tz_localize(None)
                    
                    # Clean and process the data - fix for index issues
                    stock_data = stock_data[~stock_data.index.duplicated(keep='first')]
                    
                    # Ensure index is datetime (without timezone)
                    stock_data.index = pd.to_datetime(stock_data.index)
                    
                    # Convert datetime index to date objects for consistency
                    stock_data.index = [idx.date() for idx in stock_data.index]
                    
                    # Check for Close column and add to dataframe
                    if 'close' in stock_data.columns:
                        # Normalize column names (yahooquery uses lowercase)
                        stock_data.rename(columns={'close': 'Close'}, inplace=True)
                    
                    if 'Close' in stock_data.columns:
                        first_date = stock_data.index.min()
                        last_date = stock_data.index.max()
                        st.write(f"{ticker}: First available date: {first_date}, Last available date: {last_date}")
            
                        stocks_df[ticker] = stock_data['Close']
                        successful_tickers.append(ticker)
                        break  # Success, exit retry loop
                    elif use_dummy_data:
                        # Generate dummy data if Close column is missing
                        stock_data = generate_dummy_stock_data(ticker, start_date, end_date)
                        stocks_df[ticker] = stock_data['Close']
                        successful_tickers.append(ticker)
                        st.warning(f"Using dummy data for {ticker} - missing Close column")
                        break
                    else:
                        if attempt < max_retries - 1:
                            time.sleep(3)
                            continue
                        else:
                            error_message = f"No 'Close' column found for {ticker} after {max_retries} attempts"
                            st.error(error_message)
                            
                except Exception as e:
                    if attempt < max_retries - 1:
                        st.warning(f"Retry {attempt+1}/{max_retries} for {ticker}: {e}")
                        time.sleep(3)
                    elif use_dummy_data:
                        # Generate dummy data if exception occurs
                        stock_data = generate_dummy_stock_data(ticker, start_date, end_date)
                        stocks_df[ticker] = stock_data['Close']
                        successful_tickers.append(ticker)
                        st.warning(f"Using dummy data for {ticker} after error: {e}")
                        break
                    else:
                        error_message = f"Error loading data for {ticker}: {e}"
                        st.error(error_message)
                        
        except Exception as e:
            if use_dummy_data:
                # Generate dummy data for unexpected errors
                stock_data = generate_dummy_stock_data(ticker, start_date, end_date)
                stocks_df[ticker] = stock_data['Close']
                successful_tickers.append(ticker)
                st.warning(f"Using dummy data for {ticker} after unexpected error: {e}")
            else:
                error_message = f"Unexpected error processing {ticker}: {e}"
                st.error(error_message)
    
    # Check if we have enough data
    if len(successful_tickers) < 2:
        st.error("Not enough tickers with valid data. Please select more tickers.")
        return None, successful_tickers
    
    # Filter by date range
    if not stocks_df.empty:
        filtered_df = stocks_df[(stocks_df.index >= start_date) & (stocks_df.index <= end_date)]
        
        if filtered_df.empty:
            st.error("No data available for the selected date range.")
            return None, successful_tickers
        
        return filtered_df, successful_tickers
    
    return None, successful_tickers

# =============================================================================
# TECH SMART BETA SYSTEM_Iwen
# =============================================================================

# TECH STOCKS ORGANIZED BY SMART BETA FACTORS
TECH_SMART_BETA_FACTORS = {
    "Tech Value Factor": {
        "ORCL": {"name": "Oracle Corporation", "rationale": "Mature tech with stable cash flows, lower valuation"},
        "IBM": {"name": "International Business Machines", "rationale": "Traditional tech value play, dividend yield"},
        "CSCO": {"name": "Cisco Systems Inc.", "rationale": "Networking leader with reasonable valuation"},
        "INTC": {"name": "Intel Corporation", "rationale": "Semiconductor value play, dividend yield"},
        "HPQ": {"name": "HP Inc.", "rationale": "Hardware value play with consistent returns"},
        "QCOM": {"name": "QUALCOMM Incorporated", "rationale": "5G leader at reasonable valuation"}
    },
    
    "Tech Growth Factor": {
        "NVDA": {"name": "NVIDIA Corporation", "rationale": "AI/GPU growth leader, high revenue growth"},
        "TSLA": {"name": "Tesla Inc.", "rationale": "EV/Energy growth story, disruptive innovation"},
        "SHOP": {"name": "Shopify Inc.", "rationale": "E-commerce platform growth, expanding TAM"},
        "SNOW": {"name": "Snowflake Inc.", "rationale": "Cloud data platform, high growth rates"},
        "PLTR": {"name": "Palantir Technologies", "rationale": "Data analytics growth, government contracts"},
        "ZM": {"name": "Zoom Video Communications", "rationale": "Video communication growth story"}
    },
    
    "Tech Quality Factor": {
        "AAPL": {"name": "Apple Inc.", "rationale": "Exceptional margins, brand moat, consistent profitability"},
        "MSFT": {"name": "Microsoft Corporation", "rationale": "Diversified revenue, strong cloud growth, stable earnings"},
        "GOOGL": {"name": "Alphabet Inc.", "rationale": "Search dominance, strong cash generation, R&D leadership"},
        "ADBE": {"name": "Adobe Inc.", "rationale": "Software moat, subscription model, consistent growth"},
        "CRM": {"name": "Salesforce Inc.", "rationale": "CRM leader, strong recurring revenue, market dominance"},
        "NOW": {"name": "ServiceNow Inc.", "rationale": "Enterprise software leader, high-quality growth"}
    },
    
    "Tech Momentum Factor": {
        "META": {"name": "Meta Platforms Inc.", "rationale": "Social media rebound, metaverse investments"},
        "AMD": {"name": "Advanced Micro Devices", "rationale": "CPU/GPU market share gains vs Intel/NVIDIA"},
        "NFLX": {"name": "Netflix Inc.", "rationale": "Streaming leader with global expansion"},
        "XYZ": {"name": "Block Inc.", "rationale": "Fintech innovation, crypto exposure"},
        "ROKU": {"name": "Roku Inc.", "rationale": "Streaming platform momentum, advertising growth"},
        "TWLO": {"name": "Twilio Inc.", "rationale": "Communication APIs, developer platform growth"}
    },
    
    "Tech Low Volatility Factor": {
        "MSFT": {"name": "Microsoft Corporation", "rationale": "Diversified revenue streams, enterprise stability"},
        "AAPL": {"name": "Apple Inc.", "rationale": "Consumer staple-like characteristics, loyal base"},
        "ORCL": {"name": "Oracle Corporation", "rationale": "Stable enterprise software, predictable revenue"},
        "IBM": {"name": "International Business Machines", "rationale": "Mature tech with lower volatility"},
        "CSCO": {"name": "Cisco Systems Inc.", "rationale": "Defensive tech, networking infrastructure"},
        "INTU": {"name": "Intuit Inc.", "rationale": "Tax/accounting software, seasonal stability"}
    },
    
    "Tech Profitability Factor": {
        "AAPL": {"name": "Apple Inc.", "rationale": "Industry-leading profit margins, premium pricing"},
        "MSFT": {"name": "Microsoft Corporation", "rationale": "Software margins, cloud profitability"},
        "GOOGL": {"name": "Alphabet Inc.", "rationale": "Search advertising margins, platform economics"},
        "META": {"name": "Meta Platforms Inc.", "rationale": "Social media monetization, network effects"},
        "ADBE": {"name": "Adobe Inc.", "rationale": "Software subscription margins, pricing power"},
        "CRM": {"name": "Salesforce Inc.", "rationale": "SaaS model profitability, customer retention"}
    }
}

# TECH SMART BETA STRATEGIES
TECH_SMART_BETA_STRATEGIES = {
    "Tech Value Recovery": {
        "factors": ["Tech Value Factor"],
        "description": "Undervalued tech companies poised for recovery or rerating",
        "risk_level": "Moderate",
        "academic_basis": "Fama-French Value Factor applied to technology sector",
        "sector_thesis": "Mature tech companies often undervalued relative to growth peers",
        "expected_performance": "Outperform during market rotations from growth to value"
    },
    
    "Tech Growth Premium": {
        "factors": ["Tech Growth Factor"],
        "description": "High-growth tech companies driving innovation and market expansion",
        "risk_level": "High", 
        "academic_basis": "Growth factor research with sector concentration",
        "sector_thesis": "Technology sector offers highest growth potential in modern economy",
        "expected_performance": "High returns in bull markets, vulnerable in downturns"
    },
    
    "Tech Quality Core": {
        "factors": ["Tech Quality Factor"],
        "description": "High-quality tech companies with sustainable competitive advantages",
        "risk_level": "Moderate-Low",
        "academic_basis": "Novy-Marx Quality Factor research",
        "sector_thesis": "Quality tech companies have durable moats and pricing power",
        "expected_performance": "Consistent outperformance with lower volatility"
    },
    
    "Tech GARP (Growth at Reasonable Price)": {
        "factors": ["Tech Growth Factor", "Tech Value Factor"],
        "description": "Growth companies trading at reasonable valuations",
        "risk_level": "Moderate-High",
        "academic_basis": "Combined Growth and Value factor research",
        "sector_thesis": "Best of both worlds - growth potential without excessive valuations",
        "expected_performance": "Strong risk-adjusted returns across market cycles"
    },
    
    "Balanced Tech Factors": {
        "factors": ["Tech Quality Factor", "Tech Value Factor", "Tech Profitability Factor"],
        "description": "Multi-factor approach balancing quality, value, and profitability in tech",
        "risk_level": "Moderate",
        "academic_basis": "Multi-factor diversification research",
        "sector_thesis": "Factor diversification reduces single-factor risk within tech sector",
        "expected_performance": "Smoother returns with factor diversification benefits"
    }
}

def calculate_tech_factor_scores(tickers, start_date, end_date):
    """Calculate factor scores for given tickers using the working data method"""
    try:
        st.write("🔍 **Debug: Starting factor score calculation...**")
        st.write(f"Tickers: {tickers}")
        st.write(f"Date range: {start_date} to {end_date}")
        
        # Check if we have the required package
        try:
            from factor_analyzer import FactorAnalyzer
            st.write("✅ FactorAnalyzer imported successfully")
        except ImportError as e:
            st.error(f"❌ Missing package: {e}")
            st.error("Please run: pip install factor-analyzer")
            return None
        
        # Use the working load_portfolio_data function instead
        st.write("📊 **Using proven data loading method...**")
        
        # Get data using the method that works for the main portfolio
        stocks_df, successful_tickers = load_portfolio_data(tickers, start_date, end_date)
        
        if stocks_df is None or len(successful_tickers) < 3:
            st.warning(f"⚠️ Could not get enough stock data. Got {len(successful_tickers) if successful_tickers else 0} stocks")
            return None
        
        st.write(f"✅ **Successfully loaded data for {len(successful_tickers)} stocks**")
        st.write(f"Stocks: {successful_tickers}")
        
        # Calculate financial metrics from the price data
        factor_data = {}
        
        for ticker in successful_tickers:
            if ticker in stocks_df.columns:
                price_series = stocks_df[ticker].dropna()
                
                if len(price_series) > 21:  # Need enough data for calculations
                    # Calculate returns
                    returns = price_series.pct_change().dropna()
                    
                    # Calculate metrics
                    volatility = returns.std() * np.sqrt(252)  # Annualized volatility
                    momentum = (price_series.iloc[-1] / price_series.iloc[-21] - 1)  # 1-month momentum
                    
                    # Calculate trend (slope of price over time)
                    x = np.arange(len(price_series))
                    slope, _ = np.polyfit(x, price_series.values, 1)
                    trend = slope / price_series.mean()  # Normalized trend
                    
                    # Mean reversion (current vs average)
                    mean_price = price_series.mean()
                    current_price = price_series.iloc[-1]
                    mean_reversion = (current_price - mean_price) / mean_price
                    
                    # Size proxy (use price as a simple proxy)
                    size_factor = np.log(current_price)
                    
                    factor_data[ticker] = {
                        'volatility': volatility,
                        'momentum': momentum,
                        'trend': trend,
                        'mean_reversion': mean_reversion,
                        'size': size_factor,
                        'returns_mean': returns.mean() * 252  # Annualized
                    }
                    
                    st.write(f"✅ {ticker}: Volatility={volatility:.3f}, Momentum={momentum:.3f}")
        
        if len(factor_data) < 3:
            st.warning(f"⚠️ Need at least 3 stocks for factor analysis, got {len(factor_data)}")
            return None
        
        # Convert to DataFrame
        factor_df = pd.DataFrame(factor_data).T
        
        st.write("🧮 **Factor DataFrame:**")
        st.dataframe(factor_df)
        
        # Handle missing values
        if factor_df.isnull().any().any():
            st.warning("⚠️ Found NaN values, filling with median...")
            factor_df = factor_df.fillna(factor_df.median())
        
        if np.isinf(factor_df.values).any():
            st.warning("⚠️ Found infinite values, replacing...")
            factor_df = factor_df.replace([np.inf, -np.inf], np.nan).fillna(factor_df.median())
        
        # Standardize the data
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        factor_df_scaled = pd.DataFrame(
            scaler.fit_transform(factor_df),
            index=factor_df.index,
            columns=factor_df.columns
        )
        
        st.write("🎯 **Performing Factor Analysis...**")
        
        try:
            # Use appropriate number of factors
            n_factors = min(3, len(factor_df.columns) - 1)
            fa = FactorAnalyzer(n_factors=n_factors, rotation='varimax')
            fa.fit(factor_df_scaled)
            
            # Get factor loadings
            loadings = pd.DataFrame(
                fa.loadings_,
                index=factor_df.columns,
                columns=[f'Factor {i+1}' for i in range(n_factors)]
            )
            
            st.write("✅ **Factor Analysis Successful!**")
            st.write("📊 **Factor Loadings:**")
            st.dataframe(loadings)
            
            # Calculate factor scores for each stock
            factor_scores = fa.transform(factor_df_scaled)
            scores_df = pd.DataFrame(
                factor_scores,
                index=factor_df.index,
                columns=[f'Factor {i+1}' for i in range(n_factors)]
            )
            
            return {
                'factor_scores': scores_df,
                'factor_loadings': loadings,
                'original_data': factor_df,
                'stocks_data': stocks_df,
                'success': True
            }
            
        except Exception as fa_error:
            st.error(f"❌ Factor Analysis failed: {fa_error}")
            st.write("🔄 **Using correlation analysis instead...**")
            
            correlation_matrix = factor_df.corr()
            
            return {
                'correlation_matrix': correlation_matrix,
                'original_data': factor_df,
                'stocks_data': stocks_df,
                'success': False,
                'fallback': True
            }
        
    except Exception as e:
        st.error(f"❌ **Error in calculate_tech_factor_scores**: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None

def create_tech_factor_heatmap(factor_scores):
    """Create factor exposure heatmap with proper error handling"""
    try:
        # Check what type of data we received
        if factor_scores is None:
            st.error("No factor scores provided")
            return None
            
        # Handle different data structures
        if isinstance(factor_scores, dict):
            if 'factor_scores' in factor_scores:
                # We have the full result dictionary from calculate_tech_factor_scores
                scores_df = factor_scores['factor_scores']
            elif 'original_data' in factor_scores:
                # Fallback to original data if factor analysis failed
                scores_df = factor_scores['original_data']
            else:
                st.warning("Unexpected factor_scores structure, using raw data")
                # Try to convert the dict directly
                scores_df = pd.DataFrame(factor_scores)
        elif isinstance(factor_scores, pd.DataFrame):
            scores_df = factor_scores
        else:
            st.error(f"Unexpected factor_scores type: {type(factor_scores)}")
            return None
        
        st.write(f"📊 **Heatmap data shape**: {scores_df.shape}")
        st.write("**Heatmap data preview:**")
        st.dataframe(scores_df.head())
        
        # Create the heatmap
        fig = px.imshow(
            scores_df.T,  # Transpose so factors are on y-axis, stocks on x-axis
            labels=dict(x="Stocks", y="Factors", color="Factor Score"),
            title="Tech Factor Exposure Heatmap",
            color_continuous_scale="RdBu_r",
            aspect="auto"
        )
        
        # Update layout for better readability
        fig.update_layout(
            width=800,
            height=400,
            title_font_size=16,
            xaxis_title="Stocks",
            yaxis_title="Factors"
        )
        
        # Rotate x-axis labels if there are many stocks
        if len(scores_df.index) > 6:
            fig.update_xaxes(tickangle=45)
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating heatmap: {e}")
        st.write("**Debug info:**")
        st.write(f"factor_scores type: {type(factor_scores)}")
        if hasattr(factor_scores, 'keys'):
            st.write(f"factor_scores keys: {list(factor_scores.keys())}")
        
        # Create a simple fallback plot
        st.write("Creating fallback visualization...")
        try:
            if isinstance(factor_scores, dict) and 'original_data' in factor_scores:
                fallback_data = factor_scores['original_data']
                fig = px.imshow(
                    fallback_data.T,
                    title="Factor Data Heatmap (Fallback)",
                    color_continuous_scale="RdBu_r"
                )
                return fig
            else:
                return None
        except:
            return None

def create_tech_smart_beta_selector():
    """Create tech-focused smart beta strategy selection interface"""
    st.header("💻 Tech Smart Beta Strategies")
    st.write("Apply academic factor investing principles to technology sector stocks")
    
    # Educational content specific to tech factors
    with st.expander("📚 Tech Sector Factor Investing", expanded=False):
        st.write("""
        **Why Tech-Focused Factor Investing?**
        
        **Academic Research Applied to Tech:**
        - **Value in Tech**: Mature tech companies with stable cash flows (ORCL, IBM, CSCO)
        - **Growth Premium**: High-growth disruptors (NVDA, TSLA, SNOW)
        - **Quality Focus**: Companies with strong moats and margins (AAPL, MSFT, GOOGL)
        - **Momentum Strategies**: Tech stocks show strong trend persistence
        - **Profitability**: Software companies with exceptional margins
        
        **Academic Papers Implemented:**
        - Fama & French (1992): "Cross-Section of Expected Stock Returns"
        - Jegadeesh & Titman (1993): "Returns to Buying Winners and Selling Losers"
        - Novy-Marx (2013): "The Other Side of Value: Quality and Return"
        """)
    
    # Strategy selection
    selected_stocks = []
    strategy_info = {}
    
    st.subheader("🎯 Choose Your Tech Factor Strategy")
    
    for strategy_name, strategy_data in TECH_SMART_BETA_STRATEGIES.items():
        with st.expander(f"📈 {strategy_name}", expanded=False):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write(f"**Strategy:** {strategy_data['description']}")
                st.write(f"**Academic Basis:** {strategy_data['academic_basis']}")
                st.write(f"**Sector Thesis:** {strategy_data['sector_thesis']}")
                
                # Show stocks in this strategy
                strategy_stocks = []
                for factor in strategy_data['factors']:
                    if factor in TECH_SMART_BETA_FACTORS:
                        factor_stocks = list(TECH_SMART_BETA_FACTORS[factor].keys())
                        strategy_stocks.extend(factor_stocks)
                
                # Remove duplicates while preserving order
                strategy_stocks = list(dict.fromkeys(strategy_stocks))
                st.write(f"**Stocks ({len(strategy_stocks)}):** {', '.join(strategy_stocks)}")
            
            with col2:
                # Risk level indicator
                risk_colors = {
                    "Low": "green", 
                    "Moderate": "orange", 
                    "Moderate-Low": "yellowgreen",
                    "Moderate-High": "darkorange", 
                    "High": "red"
                }
                risk_color = risk_colors.get(strategy_data['risk_level'], 'gray')
                
                st.markdown(f"<div style='text-align: center; color: {risk_color}; font-weight: bold; font-size: 16px; padding: 10px; border: 2px solid {risk_color}; border-radius: 10px;'>{strategy_data['risk_level']} Risk</div>", 
                           unsafe_allow_html=True)
                
                if st.button(f"Select Strategy", key=f"tech_strategy_{strategy_name}"):
                    selected_stocks = strategy_stocks
                    strategy_info = strategy_data
                    st.success(f"Selected {strategy_name}")
                    st.session_state.portfolio_tickers = selected_stocks
    
    return selected_stocks, strategy_info



def display_tech_factor_results(price_data, factor_scores, strategy_info=None):
    """Display factor analysis results with proper error handling"""
    try:
        st.subheader("**Tech Factor Analysis Results**")
        
        # Handle strategy_info safely
        if strategy_info:
            if isinstance(strategy_info, dict):
                strategy_name = strategy_info.get('name', strategy_info.get('title', 'Smart Beta Tech Strategy'))
                strategy_desc = strategy_info.get('description', strategy_info.get('desc', 'Factor-based investment in high-quality technology companies'))
            else:
                strategy_name = str(strategy_info)
                strategy_desc = "No description available"
        else:
            strategy_name = "Smart Beta Tech Strategy"
            strategy_desc = "Factor-based investment in high-quality technology companies"

        st.write(f"**Strategy:** {strategy_name}")
        st.write(f"**Sector Thesis:** {strategy_desc}")
        
        
        if isinstance(factor_scores, dict):
            st.write(f"factor_scores keys: {list(factor_scores.keys())}")
            
        if price_data is not None:
            if hasattr(price_data, 'shape'):
                st.write(f"price_data shape: {price_data.shape}")
                st.write("✅ price_data has data!")
            else:
                st.write(f"price_data type: {type(price_data)}")
        else:
            st.write("❌ price_data is None")
        
        # Handle the factor_scores data structure properly
        if factor_scores is None:
            st.error("No factor analysis results available")
            return
        
        # Extract the right DataFrame based on what we have
        if isinstance(factor_scores, dict):
            if 'factor_scores' in factor_scores:
                factor_df = factor_scores['factor_scores']
                st.write("✅ Using factor_scores from analysis")
            elif 'original_data' in factor_scores:
                factor_df = factor_scores['original_data']
                st.write("✅ Using original_data as fallback")
            elif 'correlation_matrix' in factor_scores:
                factor_df = factor_scores['correlation_matrix']
                st.write("✅ Using correlation_matrix as fallback")
            else:
                st.error("Could not find usable data in factor_scores")
                return
        elif isinstance(factor_scores, pd.DataFrame):
            factor_df = factor_scores
            st.write("✅ Using factor_scores DataFrame directly")
        else:
            st.error(f"Unexpected factor_scores type: {type(factor_scores)}")
            return
        
        # Display the data
        st.write("📊 **Factor Data:**")
        st.dataframe(factor_df)
        
        # Create and display heatmap
        st.subheader("**Factor Exposure Heatmap:**")
        try:
            fig_heatmap = create_tech_factor_heatmap(factor_scores)
            if fig_heatmap:
                st.plotly_chart(fig_heatmap, use_container_width=True)
            else:
                st.warning("Could not create heatmap")
        except Exception as e:
            st.error(f"Error creating heatmap: {e}")
        
        # Display factor loadings if available
        if isinstance(factor_scores, dict) and 'factor_loadings' in factor_scores:
            st.subheader("**Factor Loadings:**")
            st.dataframe(factor_scores['factor_loadings'])
        
        # Create portfolio optimization section
        st.subheader("**Portfolio Optimization Results:**")
        
        # DETAILED DEBUG (can be delteted)
        st.write("🔍 **DETAILED DEBUG FOR PORTFOLIO OPTIMIZATION:**")
        st.write(f"price_data is None: {price_data is None}")
        if price_data is not None:
            st.write(f"price_data type: {type(price_data)}")
            st.write(f"price_data has shape attr: {hasattr(price_data, 'shape')}")
            if hasattr(price_data, 'shape'):
                st.write(f"price_data shape: {price_data.shape}")
                st.write(f"price_data empty: {price_data.empty if hasattr(price_data, 'empty') else 'no empty attr'}")

        if isinstance(factor_scores, dict):
            st.write(f"'stocks_data' in factor_scores: {'stocks_data' in factor_scores}")
            if 'stocks_data' in factor_scores:
                st.write(f"factor_scores['stocks_data'] type: {type(factor_scores['stocks_data'])}")
        
        # Get stock data from multiple sources
        stocks_data = None
        
        # First, try the price_data parameter
        if price_data is not None and hasattr(price_data, 'shape'):
            stocks_data = price_data
            st.write("✅ Using price_data parameter")
        # Then try stocks_data from factor_scores
        elif isinstance(factor_scores, dict) and 'stocks_data' in factor_scores:
            stocks_data = factor_scores['stocks_data']
            st.write("✅ Using stocks_data from factor_scores")
        else:
            st.write("❌ No stock data found in any source")
        # ADD DEBUG SECTION HERE (only if stocks_data exists):
        if stocks_data is not None:
            st.write(f"📊 Stock data shape: {stocks_data.shape}")
            st.write("Stock columns:", list(stocks_data.columns))
            st.write("Stock data date range:")
            st.write(f"  - Start: {stocks_data.index[0]}")
            st.write(f"  - End: {stocks_data.index[-1]}")
            st.write("Sample expected returns:")
            mu = expected_returns.mean_historical_return(stocks_data)
            st.write(mu.head())
        
        if stocks_data is not None and hasattr(stocks_data, 'empty') and not stocks_data.empty:
            try:
                st.write(f"📊 Stock data shape: {stocks_data.shape}")
                st.write("Stock columns:", list(stocks_data.columns))
                
                # Use the stocks data for optimization
                mu = expected_returns.mean_historical_return(stocks_data)
                S = risk_models.sample_cov(stocks_data)
                
                # Optimize portfolio
                ef = EfficientFrontier(mu, S)
                risk_free_rate = 0.02
                ef.max_sharpe(risk_free_rate=risk_free_rate)
                weights = ef.clean_weights()
                expected_annual_return, annual_volatility, sharpe_ratio = ef.portfolio_performance(risk_free_rate=risk_free_rate)
                
                # Display results
                weights_df = pd.DataFrame.from_dict(weights, orient='index', columns=['Weight'])
                st.dataframe(weights_df)
                
                st.write(f"**Expected Annual Return:** {expected_annual_return:.2%}")
                st.write(f"**Annual Volatility:** {annual_volatility:.2%}")
                st.write(f"**Sharpe Ratio:** {sharpe_ratio:.2f}")
                
            except Exception as e:
                st.error(f"Error in portfolio optimization: {e}")
                import traceback
                st.code(traceback.format_exc())
        else:
            st.warning("No stock data available for portfolio optimization")
    
    except Exception as e:
        st.error(f"Error in display_tech_factor_results: {e}")
        import traceback
        st.code(traceback.format_exc())

def enhanced_stock_selection():
    """Enhanced stock selection interface for manual picking"""
    
    # Tech stock categories for easier selection
    tech_categories = {
        "💻 Big Tech": ["AAPL", "MSFT", "GOOGL", "AMZN", "META"],
        "🚀 Growth Tech": ["NVDA", "TSLA", "SHOP", "SNOW", "PLTR", "ZM"],
        "🏢 Enterprise Tech": ["CRM", "NOW", "ADBE", "ORCL", "IBM"],
        "💰 Value Tech": ["INTC", "CSCO", "HPQ", "QCOM"],
        "🎮 Consumer Tech": ["NFLX", "ROKU", "XYZ", "TWLO", "AMD"],
        "☁️ Cloud & Software": ["MSFT", "GOOGL", "AMZN", "CRM", "NOW", "SNOW", "ADBE"]
    }
    
    st.write("### 📊 Manual Tech Stock Selection")
    st.write("Choose individual stocks for your portfolio from curated tech categories")
    
    selected_stocks = []
    
    # Method 1: Category-based selection
    st.write("**Method 1: Select by Category**")
    
    for category, stocks in tech_categories.items():
        with st.expander(f"{category} ({len(stocks)} stocks)", expanded=False):
            cols = st.columns(min(len(stocks), 4))
            for i, stock in enumerate(stocks):
                with cols[i % 4]:
                    if st.checkbox(stock, key=f"cat_{category}_{stock}"):
                        if stock not in selected_stocks:
                            selected_stocks.append(stock)
    
    st.write("---")
    
    # Method 2: Direct ticker input
    st.write("**Method 2: Enter Tickers Directly**")
    custom_tickers = st.text_input(
        "Enter stock tickers (comma-separated):",
        placeholder="e.g., AAPL, MSFT, GOOGL",
        help="Enter stock ticker symbols separated by commas"
    )
    
    if custom_tickers:
        # Parse and clean the input
        custom_list = [ticker.strip().upper() for ticker in custom_tickers.split(",") if ticker.strip()]
        for ticker in custom_list:
            if ticker not in selected_stocks and len(ticker) <= 5:  # Basic validation
                selected_stocks.append(ticker)
    
    st.write("---")
    
    # Method 3: Popular presets
    st.write("**Method 3: Quick Presets**")
    preset_cols = st.columns(3)
    
    with preset_cols[0]:
        if st.button("🏆 FAANG Portfolio"):
            preset_stocks = ["META", "AAPL", "AMZN", "NFLX", "GOOGL"]
            for stock in preset_stocks:
                if stock not in selected_stocks:
                    selected_stocks.append(stock)
    
    with preset_cols[1]:
        if st.button("🚀 AI/ML Focus"):
            preset_stocks = ["NVDA", "GOOGL", "MSFT", "PLTR", "AMD"]
            for stock in preset_stocks:
                if stock not in selected_stocks:
                    selected_stocks.append(stock)
    
    with preset_cols[2]:
        if st.button("💼 Enterprise Tech"):
            preset_stocks = ["MSFT", "CRM", "NOW", "ORCL", "ADBE"]
            for stock in preset_stocks:
                if stock not in selected_stocks:
                    selected_stocks.append(stock)
    
    # Display current selection
    if selected_stocks:
        st.write("---")
        st.write("### 🎯 Your Selected Stocks:")
        
        # Display in a nice format
        selection_cols = st.columns(min(len(selected_stocks), 6))
        for i, stock in enumerate(selected_stocks):
            with selection_cols[i % 6]:
                st.success(f"**{stock}**")
        
        # Option to remove stocks
        st.write("**Remove stocks:**")
        remove_cols = st.columns(min(len(selected_stocks), 6))
        stocks_to_remove = []
        
        for i, stock in enumerate(selected_stocks):
            with remove_cols[i % 6]:
                if st.button(f"❌", key=f"remove_{stock}", help=f"Remove {stock}"):
                    stocks_to_remove.append(stock)
        
        # Remove selected stocks
        for stock in stocks_to_remove:
            if stock in selected_stocks:
                selected_stocks.remove(stock)
                st.rerun()
        
        st.write(f"**Total Selected:** {len(selected_stocks)} stocks")
        
        if len(selected_stocks) < 2:
            st.warning("⚠️ Select at least 2 stocks for portfolio optimization")
        elif len(selected_stocks) > 20:
            st.warning("⚠️ Consider reducing to 10-15 stocks for better optimization")
        else:
            st.success(f"✅ Good selection! {len(selected_stocks)} stocks ready for optimization")
    
    else:
        st.info("👆 Select stocks using any of the methods above")
    
    return selected_stocks

# Alternative simpler version if you prefer
def simple_stock_selection():
    """Simple stock selection interface"""
    st.write("### 📊 Select Tech Stocks")
    
    # Predefined list of popular tech stocks
    all_tech_stocks = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", 
        "NFLX", "ADBE", "CRM", "ORCL", "INTC", "AMD", "CSCO",
        "NOW", "SHOP", "SNOW", "PLTR", "ZM", "XYZ", "ROKU"
    ]
    
    selected_stocks = st.multiselect(
        "Choose stocks for your portfolio:",
        options=all_tech_stocks,
        default=[],
        help="Select 2 or more stocks for portfolio optimization"
    )
    
    # Option to add custom tickers
    custom_input = st.text_input(
        "Or add custom tickers (comma-separated):",
        placeholder="e.g., UBER, LYFT, SPOT"
    )
    
    if custom_input:
        custom_tickers = [ticker.strip().upper() for ticker in custom_input.split(",")]
        for ticker in custom_tickers:
            if ticker and ticker not in selected_stocks:
                selected_stocks.append(ticker)
    
    if len(selected_stocks) < 2:
        st.warning("Please select at least 2 stocks")
    
    return selected_stocks
    
def integrate_tech_smart_beta():
    """Main integration function - replaces the Portfolio Optimizer tab content"""
    st.header("🎯 Tech Smart Beta Portfolio Optimizer")
    st.write("Apply academic factor investing principles to technology stocks")
    
    # Date selection
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", datetime(2020, 1, 1))
    with col2:
        end_date = st.date_input("End Date", datetime(2024, 1, 1))
    
    # Strategy selection
    investment_approach = st.radio(
        "Choose your investment approach:",
        [
            "🧠 Smart Beta Factor Strategies (Academic Approach)",
            "📊 Traditional Tech Selection (Manual Picking)"
        ],
        help="Smart Beta applies academic factor research to systematic stock selection"
    )
    
    selected_tickers = []
    strategy_info = {}
    
    if investment_approach.startswith("🧠 Smart Beta"):
        # Smart Beta approach
        st.write("---")
        selected_tickers, strategy_info = create_tech_smart_beta_selector()
        
        # Educational content
        if selected_tickers:
            with st.expander("🎓 Academic Foundation", expanded=False):
                st.write("""
                **Your Smart Beta Strategy Demonstrates:**
                - **Factor Models**: Fama-French research applied to tech sector
                - **Portfolio Theory**: Multi-factor optimization techniques
                - **Risk Management**: Factor diversification principles
                - **Behavioral Finance**: Momentum and quality anomalies
                
                **Academic Papers Referenced:**
                - Fama & French (1992): "Cross-Section of Expected Stock Returns"
                - Jegadeesh & Titman (1993): "Returns to Buying Winners and Selling Losers"
                - Novy-Marx (2013): "The Other Side of Value: Quality and Return"
                """)
    
    else:
        # Traditional approach (your existing system)
        st.write("---")
        selected_tickers = enhanced_stock_selection()
        strategy_info = {'description': 'Traditional tech stock selection'}
    
    # Portfolio Analysis
    if selected_tickers and len(selected_tickers) >= 2:
        # Store in session state
        st.session_state.portfolio_tickers = selected_tickers
        
        # Load portfolio data
        stocks_df, successful_tickers = load_portfolio_data(selected_tickers, start_date, end_date)
        
        # DEBUG THE CONDITION
        st.write("🔍 **Debug - Optimization Condition Check:**")
        st.write(f"stocks_df is not None: {stocks_df is not None}")
        if stocks_df is not None:
            st.write(f"stocks_df is not empty: {not stocks_df.empty}")
            st.write(f"stocks_df shape: {stocks_df.shape}")
        st.write(f"successful_tickers exists: {'successful_tickers' in locals()}")
        if 'successful_tickers' in locals():
            st.write(f"successful_tickers: {successful_tickers}")
            st.write(f"len(successful_tickers): {len(successful_tickers)}")
            st.write(f"len(successful_tickers) >= 2: {len(successful_tickers) >= 2}")
        else:
            st.write("❌ successful_tickers not defined")
        
        if stocks_df is not None and not stocks_df.empty and len(stocks_df.columns) >= 2:
            st.write("---")
            st.header("📊 Portfolio Analysis & Optimization")
            
            # Factor Analysis (for Smart Beta approach)
            if investment_approach.startswith("🧠 Smart Beta"):
                st.write("### 🔍 Factor Analysis")
                with st.spinner("Calculating factor exposures..."):
                    # Fetch detailed stock data for factor analysis
                    detailed_stock_data = {}
                    for ticker in successful_tickers:
                        try:
                            stock_data = fetch_stock_data(ticker, period="1y")
                            if stock_data:
                                detailed_stock_data[ticker] = stock_data
                        except:
                            continue
                    
                    # Calculate factor scores
                    factor_scores = calculate_tech_factor_scores(selected_tickers, start_date, end_date)
                    
                    # Display factor analysis
                    if factor_scores:
                        # Get stocks_df from factor_scores if needed
                        if 'stocks_df' not in locals() and isinstance(factor_scores, dict) and 'stocks_data' in factor_scores:
                            stocks_df = factor_scores['stocks_data']
    
                        # Call the function safely - ONLY ONCE
                        if 'stocks_df' in locals() and stocks_df is not None:
                            display_tech_factor_results(stocks_df, factor_scores, strategy_info)
                        elif isinstance(factor_scores, dict) and 'stocks_data' in factor_scores:
                            display_tech_factor_results(factor_scores['stocks_data'], factor_scores, strategy_info)
                        else:
                            display_tech_factor_results(None, factor_scores, strategy_info)
                    else:
                        st.warning("⚠️ Could not calculate factor scores - using basic analysis")
            
            # Portfolio Optimization
            st.write("### 🎯 Portfolio Optimization")
            
            # ADD THIS DEBUG SECTION:
            st.write("🔍 **Debug - Before Optimization:**")
            st.write(f"stocks_df type: {type(stocks_df)}")
            st.write(f"stocks_df is None: {stocks_df is None}")
            if stocks_df is not None:
                st.write(f"stocks_df shape: {stocks_df.shape}")
                st.write(f"stocks_df empty: {stocks_df.empty}")
                st.write("stocks_df columns:", list(stocks_df.columns))
            else:
                st.write("❌ stocks_df is None - optimization will not run")
            
            
            # Calculate expected returns and covariance
            mu = expected_returns.mean_historical_return(stocks_df)
            S = risk_models.sample_cov(stocks_df)
            
            # Portfolio optimization
            try:
                ef = EfficientFrontier(mu, S)
                ef.max_sharpe(risk_free_rate=0.02)
                
                weights = ef.clean_weights()
                expected_annual_return, annual_volatility, sharpe_ratio = ef.portfolio_performance(risk_free_rate=0.02)
            
            except Exception as e:
                st.warning(f"ECOS solver failed: {e}")
                st.write("Trying SCS solver...")

                try:
                    # Try with SCS solver
                    ef = EfficientFrontier(mu, S, solver='SCS')
                    ef.max_sharpe(risk_free_rate=0.02)
                    weights = ef.clean_weights()
                    expected_annual_return, annual_volatility, sharpe_ratio = ef.portfolio_performance(risk_free_rate=0.02)

                except Exception as e2:
                    st.warning(f"SCS solver failed: {e2}")
                    st.write("Trying CLARABEL solver...")

                    try:
                        # Try with CLARABEL solver (newer, handles numerical issues well)
                        ef = EfficientFrontier(mu, S, solver='CLARABEL')
                        ef.max_sharpe(risk_free_rate=0.02)
                        weights = ef.clean_weights()
                        expected_annual_return, annual_volatility, sharpe_ratio = ef.portfolio_performance(risk_free_rate=0.02)

                    except Exception as e3:
                        st.warning(f"CLARABEL solver failed: {e3}")
                        st.write("Using regularized covariance matrix with ECOS...")
            
                        try:
                            # Add regularization to the covariance matrix to fix numerical issues
                            import numpy as np
                            S_reg = S + np.eye(len(S)) * 1e-6  # Add small value to diagonal
                
                            ef = EfficientFrontier(mu, S_reg, solver='ECOS')
                            ef.max_sharpe(risk_free_rate=0.02)
                            weights = ef.clean_weights()
                            expected_annual_return, annual_volatility, sharpe_ratio = ef.portfolio_performance(risk_free_rate=0.02)
                
                        except Exception as e4:
                            st.error(f"All solvers failed. Using equal weights fallback.")
                            st.error(f"Final error: {e4}")
                
                            # Fallback to equal weights
                            tickers = stocks_df.columns
                            weights = {ticker: 1/len(tickers) for ticker in tickers}
                
                            # Calculate performance manually for equal weights
                            portfolio_returns = stocks_df.pct_change().dropna().mean(axis=1)
                            expected_annual_return = portfolio_returns.mean() * 252
                            annual_volatility = portfolio_returns.std() * np.sqrt(252)
                            sharpe_ratio = (expected_annual_return - 0.02) / annual_volatility if annual_volatility > 0 else 0
                
            # Store results in session state
            st.session_state.portfolio_weights = weights
            st.session_state.portfolio_sharpe = sharpe_ratio
            st.session_state.portfolio_volatility = annual_volatility
            st.session_state.portfolio_expected_return = expected_annual_return
                
                # Display results
            st.subheader("🎯 Optimization Results")
                
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Expected Annual Return", f"{(expected_annual_return*100):.1f}%")
            with col2:
                st.metric("Annual Volatility", f"{(annual_volatility*100):.1f}%")
            with col3:
                st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
                
            # Weights display
            weights_df = pd.DataFrame.from_dict(weights, orient='index', columns=['Weight'])
            weights_df['Weight %'] = (weights_df['Weight'] * 100).round(1)
            weights_df = weights_df.sort_values('Weight', ascending=False)
                
            col1, col2 = st.columns([1, 1])
                
            with col1:
                st.write("**Portfolio Allocation:**")
                st.dataframe(weights_df, use_container_width=True)
                
            with col2:
                # Portfolio pie chart
                fig_weights = px.pie(
                    values=weights_df['Weight'].values,
                    names=weights_df.index,
                    title="Portfolio Allocation"
                )
                fig_weights.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_weights, use_container_width=True)
                
            # Calculate portfolio performance
            stocks_df['Optimized Portfolio'] = 0
            for ticker, weight in weights.items():
                if ticker in stocks_df.columns:
                    stocks_df['Optimized Portfolio'] += stocks_df[ticker] * weight
                
            # Performance charts
            st.subheader("📈 Portfolio Performance")
                
            # Cumulative returns
            fig_cum_returns = plot_cum_returns(stocks_df['Optimized Portfolio'], 
                                                'Optimized Tech Portfolio: Cumulative Returns')
            st.plotly_chart(fig_cum_returns, use_container_width=True)
                
            # Efficient frontier
            try:
                fig_ef = plot_efficient_frontier_and_max_sharpe(mu, S)
                fig_efficient_frontier = BytesIO()
                fig_ef.savefig(fig_efficient_frontier, format="png")
                st.image(fig_efficient_frontier, caption="Efficient Frontier with Maximum Sharpe Ratio Portfolio")
            except Exception as e:
                st.warning(f"Could not generate efficient frontier plot: {e}")

            # End of portfolio optimization section

# =============================================================================
# END OF SMART BETA FUNCTIONS TO ADD
# =============================================================================

st.set_page_config(page_title = "I-Wen's Stock Portfolio Optimizer & Advisor", layout = "wide")
st.header("I-Wen's Stock Portfolio Optimizer & AI Stock Advisor")

# Set up the tabs for different features
tabs = st.tabs(["Portfolio Optimizer", "AI Stock Advisor"])

# Portfolio Optimizer tab
with tabs[0]:
    integrate_tech_smart_beta()

# AI Stock Advisor tab
with tabs[1]:
    # Sidebar with API key input in this tab
    api_key = get_openai_api_key()
    client = get_openai_client(api_key)
    
    # Main layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Chat with your Stock Advisor")
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask about stocks, investment strategies, or specific companies..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Only proceed if API key is provided
            if not client:
                with st.chat_message("assistant"):
                    st.markdown("Please provide your OpenAI API key in the sidebar to continue.")
                st.session_state.messages.append({"role": "assistant", "content": "Please provide your OpenAI API key in the sidebar to continue."})
            else:
                # Display assistant response
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        response = process_query(client, prompt)
                        st.markdown(response)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Right sidebar for visualizations if a ticker has been selected
    with col2:
        if st.session_state.current_ticker and st.session_state.ticker_data:
            ticker = st.session_state.current_ticker
            stock_data = st.session_state.ticker_data
            
            st.subheader(f"{ticker} Insights")
            
            # Stock price chart
            st.plotly_chart(get_stock_price_chart(ticker, stock_data["history"]), use_container_width=True)
            
            # Technical Indicators
            with st.expander("Technical Indicators", expanded=False):
                tech_df = calculate_technical_indicators(stock_data["history"])
                fig = plot_technical_indicators(tech_df, ticker)
                st.pyplot(fig)
            
            # Financial Metrics
            with st.expander("Financial Metrics", expanded=True):
                metrics = get_financial_metrics(stock_data)
                for key, value in metrics.items():
                    if key == "Market Cap" and isinstance(value, (int, float)):
                        value = f"${value/1e9:.2f}B" if value >= 1e9 else f"${value/1e6:.2f}M"
                    elif key == "Dividend Yield" and isinstance(value, (int, float)):
                        value = f"{value*100:.2f}%" if value else 'N/A'
                    st.metric(key, value)
            
            # Recent News
            with st.expander("Recent News", expanded=True):
                news = get_recent_news(ticker)
                for item in news:
                    st.markdown(f"• {item}")

# Hide Streamlit style
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
