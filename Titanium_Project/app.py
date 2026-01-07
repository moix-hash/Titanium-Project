import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model


st.set_page_config(page_title="Titanium Analytics", layout="wide")

st.markdown("""
<style>
    .main {background-color: #f4f4f4;}
    h1 {color: #2c3e50; font-family: 'Helvetica', sans-serif;}
    .stButton>button {background-color: #2c3e50; color: white; border-radius: 5px;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_all_models():
    try:
        lr_model = joblib.load('models/lr_model.pkl')
        stock_scaler = joblib.load('models/stock_scaler.pkl')
        lstm_model = load_model('models/lstm_model.h5')
        
        sent_model = joblib.load('models/sentiment_model.pkl')
        
        kmeans_model = joblib.load('models/kmeans_model.pkl')
        fund_scaler = joblib.load('models/fund_scaler.pkl')
        
        return lr_model, stock_scaler, lstm_model, sent_model, kmeans_model, fund_scaler
    except Exception as e:
        return None, None, None, None, None, None

lr, s_scaler, lstm, sent, km, f_scaler = load_all_models()

st.sidebar.title("TITANIUM ANALYTICS")
st.sidebar.info("Enterprise Data Suite")
page = st.sidebar.radio("Go To Module:", ["Dashboard Home", "Live Predictions", "Data Studio"])

if page == "Dashboard Home":
    st.title("Executive Dashboard")
    st.write("Overview of trained models and dataset health.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Stock Market Model")
        st.write("**Type:** Linear Regression & LSTM (Deep Learning)")
        st.write("**Status:** Active")
        st.progress(95)
    
    with col2:
        st.subheader("News Sentiment Model")
        st.write("**Type:** Naive Bayes (NLP)")
        st.write("**Status:** Active")
        st.progress(88)

    st.write("---")
    st.subheader("Data Visualizations")
  
    try:
        df_stock = pd.read_csv('processed_stock.csv')
        st.line_chart(df_stock.set_index('Date')['Close'])
    except:
        st.warning("Processed stock data not found. Please run Notebook 1.")


elif page == "Live Predictions":
    st.title("AI Prediction Lab")
    
    tab1, tab2, tab3 = st.tabs(["Stock Price", "News Sentiment", "Company Cluster"])
    
    with tab1:
        st.subheader("Predict Future Stock Price")
        c1, c2, c3 = st.columns(3)
        open_val = c1.number_input("Open Price", 100.0)
        high_val = c2.number_input("High Price", 110.0)
        low_val = c3.number_input("Low Price", 95.0)
        vol_val = c1.number_input("Volume", 5000000)
        ma_val = c2.number_input("50-Day Avg", 102.0)
        
        if st.button("Predict Price"):
            if lr:
                input_data = np.array([[open_val, high_val, low_val, vol_val, ma_val]])
                input_scaled = s_scaler.transform(input_data)
                
                
                pred_lr = lr.predict(input_scaled)[0]
                
                pred_lstm = lstm.predict(input_scaled.reshape(1,1,5), verbose=0)[0][0]
                
                st.success(f"Linear Regression Forecast: ${pred_lr:.2f}")
                st.info(f"Deep Learning Forecast: ${pred_lstm:.2f}")
            else:
                st.error("Models not found. Please run the training notebooks.")

    with tab2:
        st.subheader("Analyze News Sentiment")
        user_text = st.text_area("Enter a financial headline:", "Quarterly earnings dropped significantly.")
        
        if st.button("Analyze"):
            if sent:
                prediction = sent.predict([user_text])[0]
                st.write(f"**Predicted Sentiment:** {prediction.upper()}")
            else:
                st.error("Model not loaded.")

  
    with tab3:
        st.subheader("Classify Company Type")
        pe = st.number_input("P/E Ratio", 25.0)
        eps = st.number_input("Earnings Per Share", 2.0)
        mcap = st.number_input("Market Cap (Billions)", 50.0)
        price = st.number_input("Current Price", 100.0)
        
        if st.button("Classify"):
            if km:
                
                in_data = np.array([[price, pe, eps, mcap*1000000000]])
                in_scaled = f_scaler.transform(in_data)
                cluster = km.predict(in_scaled)[0]
                st.metric("Cluster Group", f"Group {cluster}")
            else:
                st.error("Model not loaded.")


elif page == "Data Studio":
    st.title("Data Management Studio")
    
    st.subheader("1. Upload New Data")
    uploaded = st.file_uploader("Upload CSV", type=['csv'])
    if uploaded:
        st.success("File uploaded successfully. (Analysis trigger would go here)")
        st.dataframe(pd.read_csv(uploaded).head())

    st.write("---")
    st.subheader("2. Download Assets")
    
    c1, c2 = st.columns(2)
    with c1:
        with open("models/lr_model.pkl", "rb") as f:
            st.download_button("Download Regression Model", f, "lr_model.pkl")
    with c2:
        try:
            with open("processed_stock.csv", "rb") as f:
                st.download_button("Download Processed Data", f, "stock_data_clean.csv")
        except:
            st.write("Processed data not available.")