FROM python:3.10.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Fix the seaborn-deep style issue in PyPortfolioOpt
RUN sed -i 's/plt.style.use("seaborn-deep")/plt.style.use("seaborn-v0_8-deep" if "seaborn-v0_8-deep" in plt.style.available else "default")/g' /usr/local/lib/python3.10/site-packages/pypfopt/plotting.py

# Copy the rest of the application
COPY . .

# Expose the port Streamlit runs on
EXPOSE 8501

# Command to run the application
CMD ["streamlit", "run", "portfolio_optimizer.py", "--server.port=8501", "--server.address=0.0.0.0"]
