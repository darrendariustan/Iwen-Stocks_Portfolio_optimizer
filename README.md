# Stocks_Portfolio_optimizer
# I-Wen's Stock Portfolio Optimizer & AI Stock Advisor

## 📋 Project Overview

This project is a comprehensive financial analysis and portfolio optimization application that combines **modern portfolio theory**, **smart beta factor investing**, and **AI-powered stock analysis**. The application demonstrates the practical implementation of academic finance theories with real-world data and cutting-edge AI technology.

## 🎯 Educational Objectives

### **Academic Finance Concepts Implemented:**
1. **Modern Portfolio Theory (Markowitz, 1952)**
   - Efficient frontier calculation and visualization
   - Risk-return optimization using quadratic programming
   - Portfolio diversification principles

2. **Factor Investing (Fama-French Model)**
   - Multi-factor portfolio construction
   - Value, Growth, Quality, Momentum, and Profitability factors
   - Smart Beta strategies applied to technology sector

3. **Technical Analysis**
   - Moving averages (MA20, MA50, MA200)
   - Relative Strength Index (RSI)
   - MACD (Moving Average Convergence Divergence)

4. **Risk Management**
   - Volatility calculations
   - Sharpe ratio optimization
   - Correlation analysis and diversification

## 🏗️ System Architecture

### **Technology Stack:**
- **Frontend**: Streamlit (Python web framework)
- **Data Sources**: Yahoo Finance via yahooquery API
- **AI Integration**: OpenAI GPT-4 for intelligent stock analysis
- **Financial Libraries**: PyPortfolioOpt, pandas, numpy
- **Visualization**: Plotly, matplotlib, seaborn

### **Application Structure:**
```
├── Portfolio Optimizer (Tab 1)
│   ├── Smart Beta Factor Strategies
│   ├── Traditional Stock Selection
│   ├── Portfolio Optimization Engine
│   └── Performance Visualization
│
└── AI Stock Advisor (Tab 2)
    ├── Intelligent Chat Interface
    ├── Real-time Stock Analysis
    ├── Technical Indicators
    └── News Integration
```

## 🚀 Key Features

### **1. Portfolio Optimizer**

#### **Smart Beta Factor Strategies:**
- **Tech Value Factor**: Undervalued technology companies (ORCL, IBM, CSCO)
- **Tech Growth Factor**: High-growth disruptors (NVDA, TSLA, SNOW)
- **Tech Quality Factor**: Companies with strong moats (AAPL, MSFT, GOOGL)
- **Tech Momentum Factor**: Trending technology stocks
- **Tech Profitability Factor**: High-margin software companies

#### **Academic Foundation:**
- **Fama & French (1992)**: "Cross-Section of Expected Stock Returns"
- **Jegadeesh & Titman (1993)**: "Returns to Buying Winners and Selling Losers"
- **Novy-Marx (2013)**: "The Other Side of Value: Quality and Return"

#### **Optimization Features:**
- Maximum Sharpe ratio optimization
- Efficient frontier visualization
- Multiple solver support (ECOS, SCS, CLARABEL)
- Risk-return trade-off analysis
- Cumulative returns tracking

### **2. AI Stock Advisor**

#### **Intelligent Analysis:**
- **Natural Language Processing**: Extracts stock tickers from conversational queries
- **Contextual Understanding**: Provides relevant analysis based on user intent
- **Portfolio Integration**: Considers user's selected stocks in recommendations

#### **Technical Analysis:**
- **Interactive Charts**: Line charts and candlestick visualizations
- **Technical Indicators**: RSI, MACD, Moving Averages with buy/sell signals
- **Real-time Data**: Live stock prices and financial metrics

#### **News Integration:**
- Recent news scraping and analysis
- Sentiment consideration in stock recommendations
- Market trend awareness

## 📊 Technical Implementation

### **Data Processing Pipeline:**
1. **Data Acquisition**: Yahoo Finance API with error handling and fallback mechanisms
2. **Data Cleaning**: Timezone normalization, duplicate removal, column standardization
3. **Feature Engineering**: Technical indicators, factor scores, risk metrics
4. **Optimization**: Quadratic programming for portfolio weights
5. **Visualization**: Interactive charts and performance metrics

### **Factor Analysis Engine:**
```python
# Example: Tech Factor Score Calculation
def calculate_tech_factor_scores(tickers, start_date, end_date):
    # Load stock data
    stocks_df = load_portfolio_data(tickers, start_date, end_date)
    
    # Calculate financial metrics
    factor_data = {
        'volatility': annual_volatility,
        'momentum': price_momentum,
        'trend': price_trend,
        'mean_reversion': reversion_score,
        'profitability': return_metrics
    }
    
    # Factor analysis using academic models
    fa = FactorAnalyzer(n_factors=3, rotation='varimax')
    factor_scores = fa.fit_transform(standardized_data)
    
    return factor_scores
```

### **Portfolio Optimization:**
```python
# Modern Portfolio Theory Implementation
def optimize_portfolio(mu, S):
    # Expected returns and covariance matrix
    ef = EfficientFrontier(mu, S)
    
    # Maximize Sharpe ratio
    ef.max_sharpe(risk_free_rate=0.02)
    weights = ef.clean_weights()
    
    # Calculate performance metrics
    expected_return, volatility, sharpe = ef.portfolio_performance()
    
    return weights, expected_return, volatility, sharpe
```

## 📈 Academic Value & Learning Outcomes

### **Quantitative Finance Skills:**
1. **Portfolio Theory**: Hands-on experience with Markowitz optimization
2. **Factor Models**: Implementation of academic factor research
3. **Risk Management**: Practical application of risk metrics
4. **Data Analysis**: Real-world financial data processing

### **Technology Integration:**
1. **API Integration**: Working with financial data APIs
2. **Machine Learning**: Factor analysis and pattern recognition
3. **AI Applications**: Natural language processing for finance
4. **Web Development**: Financial application deployment

### **Professional Development:**
1. **Industry Tools**: Experience with professional portfolio optimization libraries
2. **Best Practices**: Error handling, data validation, user experience design
3. **Academic Research**: Implementation of peer-reviewed financial models

## 🛠️ Installation & Setup

### **Prerequisites:**
```bash
pip install streamlit pandas numpy plotly
pip install pypfopt yahooquery openai
pip install scikit-learn factor-analyzer
pip install matplotlib seaborn beautifulsoup4
```

### **Configuration:**
1. **OpenAI API Key**: Required for AI stock advisor functionality
2. **Data Sources**: Yahoo Finance (free tier sufficient)
3. **Computing Requirements**: Standard Python environment

### **Running the Application:**
```bash
streamlit run your_app.py
```

## 📚 Educational Extensions

### **Potential Enhancements for Advanced Study:**
1. **Options Pricing**: Black-Scholes model implementation
2. **Risk Models**: VaR (Value at Risk) calculations
3. **Backtesting**: Historical strategy performance analysis
4. **Alternative Data**: Sentiment analysis, satellite data integration
5. **Cryptocurrency**: Digital asset portfolio optimization

### **Research Opportunities:**
1. **Factor Effectiveness**: Test factor performance across different market regimes
2. **AI Enhancement**: Improve prediction accuracy with advanced ML models
3. **Behavioral Finance**: Incorporate investor psychology into recommendations
4. **ESG Integration**: Environmental, Social, Governance factor inclusion

## 🎓 Academic References

### **Core Papers Implemented:**
1. **Markowitz, H. (1952)**: "Portfolio Selection", Journal of Finance
2. **Fama, E. F., & French, K. R. (1992)**: "Cross-Section of Expected Stock Returns"
3. **Jegadeesh, N., & Titman, S. (1993)**: "Returns to Buying Winners and Selling Losers"
4. **Novy-Marx, R. (2013)**: "The Other Side of Value: Quality and Return"

### **Technical Analysis References:**
1. **Murphy, J. J. (1999)**: "Technical Analysis of the Financial Markets"
2. **Lo, A. W., & MacKinlay, A. C. (1999)**: "A Non-Random Walk Down Wall Street"

## 🔮 Future Development

### **Short-term Enhancements:**
- Real-time portfolio tracking
- Advanced charting features
- Mobile responsiveness
- Performance attribution analysis

### **Long-term Vision:**
- Machine learning prediction models
- Alternative investment integration
- Social trading features
- Institutional-grade risk management

## 📞 Contact & Support

This project demonstrates the practical application of academic finance theory using modern technology. It serves as both a learning tool for quantitative finance concepts and a foundation for professional portfolio management applications.

---

**Academic Institution**: Esade Business School 
**Course**: Asset Management 
**Semester**: Term 3 
**Student**: I-Wen  

*This project showcases the integration of academic finance theory with practical application development, demonstrating both theoretical understanding and technical implementation skills.*
