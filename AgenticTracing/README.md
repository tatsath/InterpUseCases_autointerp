# Simple Finance Agents System

A streamlined system for financial analysis with specialized agents using real-time market data.

## Features

- **Market Data Analysis**: Current stock prices, financial metrics, and market data
- **News Analysis**: Recent financial news and market sentiment
- **Sector Analysis**: Sector performance and market trends
- **Investment Recommendations**: Risk assessments and portfolio insights

## Setup

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Activate conda environment**:
   ```bash
   conda activate sae
   ```

## Usage

### Quick Start
```bash
python run_finance_agents.py
```

### Direct Usage
```bash
python llama_finance_agents.py
```

### Programmatic Usage
```python
from llama_finance_agents import SimpleFinanceAgents

# Initialize agents
agents = SimpleFinanceAgents()
agents.setup()

# Analyze a stock
result = agents.analyze_stock("AAPL")
print(result)

# Analyze a sector
result = agents.analyze_sector("technology")
print(result)
```

## Available Tools

- **Stock Price Analysis**: Current prices, market cap, 52-week highs/lows
- **Financial Metrics**: P/E ratios, debt-to-equity, ROE, growth rates
- **News Analysis**: Recent news and market sentiment
- **Sector Performance**: Sector ETFs and performance metrics

## Agent Specializations

1. **Market Data Agent**: Gathers stock prices and financial metrics
2. **News Analysis Agent**: Collects and analyzes financial news
3. **Sector Analysis Agent**: Analyzes sector performance and trends
4. **Investment Advisor Agent**: Provides recommendations and risk assessments

## Example Output

The system provides comprehensive analysis including:
- Current stock metrics and performance
- Recent news and market sentiment
- Sector context and trends
- Investment recommendations with risk assessment

## Requirements

- Python 3.8+
- Internet connection for real-time data
- Yahoo Finance API access

## Notes

- All financial data is sourced from Yahoo Finance
- System provides real-time market data
- Results include proper disclaimers about investment risks
- No external API keys required for basic functionality
