# Stock Market Price Prediction
This project predicts future stock prices using LSTM (Long Short-Term Memory) neural networks, technical indicators, and historical data. It features an interactive Streamlit frontend for visualizing predictions and a chatbot assistant powered by LLMs for stock-related queries.


## Demo 

Visit [here](https://stockmarketpricepredict.streamlit.app/)
## Features
- Interactive UI with Streamlit for choosing tickers and forecast horizon

- LSTM-based forecasting model trained per stock ticker

- Technical Indicators: SMA (10, 20, 50), EMA (20) for enriched feature extraction

- Data preprocessing: normalization, cleanup, rolling mean, etc.

- Conversational Assistant using an LLM for financial queries

- Evaluation Metrics: MSE and R² Score

- Forecast Visualization: Next N-day price prediction plot

- Supports 5 tickers: BAC, JNJ, MSFT, NVDA, TATAPOWER

## How to Run

1. Install dependencies
```bash
pip install -r requirements.txt
```

2. Run the Streamlit app
```bash
streamlit run main.py
```
##  Model Overview

- Framework: Keras with TensorFlow backend

- Architecture: LSTM → Dropout → Dense

- Training Input: 60-time step sliding window on normalized indicators

- Output: 1-step ahead Close price

## Metrics Used
- MSE (Mean Squared Error)

- R² Score (Explained Variance)

## Chatbot (StockBot)
- Powered by Groq LLM through LangChain

- Supports follow-up context

- Can explain financial terms, company insights, and more