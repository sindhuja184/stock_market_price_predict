import streamlit as st
import pandas as pd
import yfinance as yf
from backend import data_preprocess, indicators, create_dataset, predict
from keras.models import load_model
from chatbot.groq_model import chain
st.title("Stock Market Price Prediction")


ticker = st.sidebar.radio(
    "Select a ticker",
    options = ['BAC', 'JNJ', 'MSFT', 'NVDA', 'TATAPOWER']
)
forecast_days = st.sidebar.slider("Days to Forecast", min_value=1, max_value=30, value=7)

if st.sidebar.button("Predict"):
    st.write(f"## Forecasting for {ticker} for next {forecast_days} days")

    with st.spinner("Fetching and preprocessing data..."):
       
        if ticker == 'BAC':
            df = pd.read_csv('CSV/BAC_data.csv')
        elif ticker == 'JNJ':
            df = pd.read_csv('CSV/JNJ_data.csv')
        elif ticker == 'MSFT':
            df = pd.read_csv('CSV/MSFT_data.csv')
        elif ticker == 'NVDA':
            df = pd.read_csv('CSV/NVDA_data.csv')
        else:
            df = pd.read_csv('CSV/TATAPOWER_data.csv')
        df.columns = ['Price' if col == 'Date' else col for col in df.columns]
        data_preprocess(df, ticker, forecast_days)
        indicators(df, forecast_days)
        time_step = 60
        X, y = create_dataset(df, time_step)
        
        st.success("Data processed successfully.")

    with st.spinner("Loading model and predicting..."):
        try:
            if ticker == 'BAC':
                model = load_model('models/model_BAC.keras')
            elif ticker == 'JNJ':
                model = load_model('models/model_JNJ.keras')
            elif ticker == 'MSFT':
                model = load_model('models/model_MSFT.keras')
            elif ticker == 'NVDA':
                model = load_model('models/model_NVDA.keras')
            else:
                model = load_model('models/model_TATAPOWER.keras')
            predict(X[int(len(X)*0.8):], y[int(len(y)*0.8):], model, forecast_days)
            st.success("Prediction complete!")
        except Exception as e:
            st.error(f"Model loading or prediction failed: {str(e)}")


st.sidebar.title("StockBot Assistant")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display previous messages
chat_container = st.sidebar.container()
with chat_container:
    for role, msg in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(msg)

# Input at the bottom of sidebar
user_input = st.sidebar.chat_input("Ask anything about stocks...")

if user_input:
    st.session_state.chat_history.append(("user", user_input))
    with chat_container:
        with st.chat_message("user"):
            st.markdown(user_input)
        with st.chat_message("assistant"):
            with st.spinner("StockBot is thinking..."):
                response = chain.invoke({"user_input": user_input})
            st.markdown(response.content)
            st.session_state.chat_history.append(("assistant", response.content))