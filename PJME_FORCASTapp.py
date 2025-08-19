# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import warnings

# --- Import all required libraries from the notebook ---
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose, STL
import pmdarima as pm
from prophet import Prophet
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, GRU, SimpleRNN
from streamlit_lottie import st_lottie

warnings.filterwarnings("ignore")

# --- Page Configuration and Cinematic Styling ---
st.set_page_config(
    page_title="PJME Power Forecasting",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Function to load Lottie animation from URL
def load_lottieurl(url: str):
    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200: return None
        return r.json()
    except requests.exceptions.RequestException:
        return None

# Advanced CSS for a cinematic, attractive look with a blue theme
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] > .main {
        background-image: linear-gradient(rgba(0, 15, 40, 0.7), rgba(0, 15, 40, 0.7)), url("https://images.alphacoders.com/133/1330023.png");
        background-size: cover; background-position: center; background-repeat: no-repeat; background-attachment: fixed;
    }
    [data-testid="stSidebar"] {
        background-color: rgba(10, 20, 35, 0.8); backdrop-filter: blur(10px); border-right: 1px solid rgba(0, 169, 255, 0.3);
    }
    h1, h2, h3 { font-weight: 700; color: #FFFFFF; text-shadow: 2px 2px 6px rgba(0, 0, 0, 0.9); }
    .custom-container { border-radius: 0.75rem; padding: 1.5rem; background: rgba(20, 30, 50, 0.75); backdrop-filter: blur(15px); box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37); margin-bottom: 2rem; border: 1px solid rgba(0, 169, 255, 0.3); }
    .st-emotion-cache-16txtl3, .st-emotion-cache-16txtl3 *, .st-emotion-cache-16txtl3 h3 { color: #FFFFFF !important; text-shadow: none; }
    div[data-baseweb="radio"] > div > label { background-color: transparent; padding: 12px 18px; margin-bottom: 8px; border-radius: 0.5rem; transition: all 0.3s ease; border: 1px solid rgba(0, 169, 255, 0.2); width: 100%; font-weight: 500; color: #E0E0E0; }
    div[data-baseweb="radio"] > div > label:hover { background-color: rgba(0, 169, 255, 0.2); color: #FFFFFF; border: 1px solid #00A9FF; transform: scale(1.02); }
    [data-testid="stAlert"] { background: rgba(0, 169, 255, 0.15); border: 1px solid rgba(0, 169, 255, 0.3); border-radius: 0.5rem; color: #FFFFFF; }
    div[data-baseweb="radio"] input[type="radio"] { display: none; }
</style>
""", unsafe_allow_html=True)

# --- Lottie Animations ---
lottie_main = load_lottieurl("https://lottie.host/17b9b219-c67d-4876-963d-2401d1d3a504/w3UaYg0DTK.json")
lottie_sarima = load_lottieurl("https://lottie.host/b049b49b-733d-4c3d-8211-1372e903a958/P1t2KjXyqW.json")
lottie_lstm = load_lottieurl("https://lottie.host/9e414279-d57f-4318-a616-16e174092f69/g92tJ1v48E.json")
lottie_arima = load_lottieurl("https://lottie.host/e3428d00-4b24-4f01-a1e6-2009214a1a54/535J7Jj8sS.json")
lottie_hw = load_lottieurl("https://lottie.host/9c336b13-75b2-4d44-a698-c920f69a19c6/c3q52k4h5n.json")
lottie_prophet = load_lottieurl("https://lottie.host/a61c7f99-232f-410a-b1a7-33633107561f/sJ7e2V3h5b.json")
lottie_gru = load_lottieurl("https://lottie.host/f8b9e6e8-d101-49b8-89c0-1011c2a0457a/6vX39sI75Z.json")
lottie_rnn = load_lottieurl("https://lottie.host/158525b4-7836-4760-b9a3-5c79e701a52e/5v4v34vI4c.json")
lottie_hybrid = load_lottieurl("https://lottie.host/d2762a93-1994-4b2e-a74c-5353846101f3/y5O4d1cGNw.json")
lottie_combo = load_lottieurl("https://lottie.host/43e74360-1506-4b25-a130-22c608f5d023/wUvR403FUD.json")
lottie_compare = load_lottieurl("https://lottie.host/65e9057b-7d92-4809-b615-58580540d9b6/b4Xg1H7v1s.json")
lottie_conclusion = load_lottieurl("https://lottie.host/e06b9968-3069-4244-9b2f-7f7253b26615/1v2iT2vV0c.json")
lottie_future = load_lottieurl("https://lottie.host/e47481c5-3a0c-4389-a78b-82a17f30a501/7xkm252IT2.json")

# --- Data Loading and Processing (Cached) ---
@st.cache_data
def load_and_process_data(file_path='PJME_hourly.csv'):
    df = pd.read_csv(file_path, parse_dates=["Datetime"])
    df = df.set_index("Datetime").sort_index()
    df_monthly = df["PJME_MW"].resample("M").mean()
    return df_monthly

df_monthly = load_and_process_data()
train_data = df_monthly.loc[:'2017-12-31']
test_data = df_monthly.loc['2018-01-31':]

# --- Helper Functions ---
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def plot_forecast(historical_data, forecast_values, model_name, color, test_data=None):
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_alpha(0.0)
    ax.set_facecolor((0, 0, 0, 0))
    ax.plot(historical_data.index, historical_data.values, label="Historical Data", color="#00A9FF", linewidth=2)
    if test_data is not None:
        ax.plot(test_data.index, test_data.values, label="Actual Test Data", color="lightgreen", linewidth=2)
    ax.plot(forecast_values.index, forecast_values, linestyle="--", label=f"{model_name} Forecast", color=color, linewidth=2)
    ax.set_title(f"PJME Monthly Forecast vs Actuals", fontsize=16, weight='bold', color='#FFFFFF')
    ax.set_xlabel("Date", color='#FFFFFF')
    ax.set_ylabel("Monthly Mean MW", color='#FFFFFF')
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('white')
    ax.legend(facecolor=(20/255, 30/255, 50/255, 0.8), edgecolor='none', labelcolor='white')
    fig.tight_layout()
    st.pyplot(fig)

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

# --- Model Training Functions (Cached) ---
@st.cache_resource
def train_sarima_model(train):
    model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    return model.fit(disp=False)

@st.cache_resource
def train_lstm_model(train):
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train.values.reshape(-1, 1))
    X_train, y_train = create_sequences(train_scaled, 12)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    model = Sequential([LSTM(50, activation="relu", input_shape=(12, 1)), Dense(1)])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X_train, y_train, epochs=200, batch_size=8, verbose=0)
    return model, scaler

@st.cache_resource
def train_arima_model(train):
    model = ARIMA(train, order=(2, 1, 2))
    return model.fit()

@st.cache_resource
def train_hw_model(train):
    model = ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods=12)
    return model.fit()

@st.cache_resource
def train_gru_model(train):
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train.values.reshape(-1, 1))
    X_train, y_train = create_sequences(train_scaled, 12)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    model = Sequential([GRU(50, activation='relu', input_shape=(12, 1)), Dense(1)])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=200, batch_size=8, verbose=0)
    return model, scaler

@st.cache_resource
def train_rnn_model(train):
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train.values.reshape(-1, 1))
    X_train, y_train = create_sequences(train_scaled, 12)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    model = Sequential([SimpleRNN(50, activation='relu', input_shape=(12, 1)), Dense(1)])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=200, batch_size=8, verbose=0)
    return model, scaler

@st.cache_resource
def train_prophet_lstm_model(train):
    prophet_df = train.reset_index().rename(columns={'PJME_MW': 'y', 'Datetime': 'ds'})
    prophet_model = Prophet()
    prophet_model.fit(prophet_df)
    forecast = prophet_model.predict(prophet_df)
    residuals = train.values - forecast['yhat'].values
    scaler = MinMaxScaler()
    residuals_scaled = scaler.fit_transform(residuals.reshape(-1, 1))
    
    X_res, y_res = create_sequences(residuals_scaled, 12)
    X_res = np.reshape(X_res, (X_res.shape[0], X_res.shape[1], 1))
    
    lstm_model = Sequential([LSTM(50, activation='relu', input_shape=(12, 1)), Dense(1)])
    lstm_model.compile(optimizer='adam', loss='mse')
    lstm_model.fit(X_res, y_res, epochs=100, batch_size=16, verbose=0)
    return prophet_model, lstm_model, scaler

@st.cache_resource
def train_tuned_arima_gru_model(train):
    stl = STL(train, seasonal=13)
    res = stl.fit()
    seasonal, trend, resid = res.seasonal, res.trend, res.resid
    
    auto_arima_model = pm.auto_arima(resid, seasonal=False, stepwise=True, suppress_warnings=True, trace=False)
    arima_order = auto_arima_model.order
    
    arima_model = ARIMA(resid, order=arima_order).fit()
    
    scaler = MinMaxScaler()
    seasonal_scaled = scaler.fit_transform(seasonal.values.reshape(-1, 1))
    
    X_seasonal, y_seasonal = create_sequences(seasonal_scaled, 12)
    X_seasonal = np.reshape(X_seasonal, (X_seasonal.shape[0], X_seasonal.shape[1], 1))
    
    gru_model = Sequential([GRU(50, activation='relu', input_shape=(12, 1)), Dense(1)])
    gru_model.compile(optimizer='adam', loss='mse')
    gru_model.fit(X_seasonal, y_seasonal, epochs=100, batch_size=16, verbose=0)
    
    return arima_model, gru_model, scaler, trend, 12

# --- Sidebar ---
st.sidebar.title("🔮 Forecasting Dashboard")
st.sidebar.markdown("Navigate to explore the forecasting models:")
page = st.sidebar.radio("Navigation",
    ["🏠 Introduction", "📈 SARIMA", "🧠 LSTM", "📊 ARIMA", "❄️ Holt-Winters", "🧠 GRU", "🤖 Simple RNN", "🔧 Prophet-LSTM Hybrid", "🔧 Tuned ARIMA-GRU", "🧩 Combination Model", "🏆 Model Comparison", "🔮 Future Forecasts", "📜 Conclusion"],
    label_visibility="collapsed")
st.sidebar.markdown("---")
st.sidebar.subheader("Tech Stack")
st.sidebar.markdown("- **🎨 Streamlit:** UI & Dashboard\n- **🐍 Pandas:** Data Manipulation\n- **📊 Matplotlib:** Plotting\n- **📈 Statsmodels:** Time-Series Models\n- **🧠 TensorFlow/Keras:** Deep Learning Models\n- **✨ Prophet & Pmdarima:** Advanced Forecasting")

# --- Main App ---
if page == "🏠 Introduction":
    st.title("⚡ PJME Power Consumption: Forecasting Models")
    st.markdown("### An interactive dashboard to visualize and compare time-series forecasting models.")
    col1, col2 = st.columns([2,1])
    with col1:
        st.markdown('<div class="custom-container">', unsafe_allow_html=True)
        st.markdown("This application presents a detailed, line-by-line analysis of various forecasting models applied to the **PJME Power Consumption** dataset. Navigate using the sidebar to explore each model's forecast, performance metrics, and a detailed interpretation from the original analysis.", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        if lottie_main: st_lottie(lottie_main, height=300, key="main")

def render_model_page(title, lottie_animation, model_name, color, forecast_function, interpretation_text):
    st.title(f"{title}")
    col1, col2 = st.columns([1, 3])
    with col1:
        if lottie_animation: st_lottie(lottie_animation, height=200, key=model_name.lower().replace(" ", ""))
    with col2:
        with st.spinner(f"Fitting {model_name} model... Please wait."):
            forecast = forecast_function()
        plot_forecast(df_monthly, forecast, model_name, color, test_data=test_data)
    st.markdown('<div class="custom-container">', unsafe_allow_html=True)
    st.subheader("Result Interpretation")
    st.info(interpretation_text, icon="💡")
    st.markdown('</div>', unsafe_allow_html=True)

# --- Forecasting Functions ---
def forecast_deep_learning(model_func, train_data, test_data):
    model, scaler = model_func(train_data)
    inputs = df_monthly[len(df_monthly) - len(test_data) - 12:].values.reshape(-1, 1)
    inputs = scaler.transform(inputs)
    X_test = []
    for i in range(12, len(inputs)):
        X_test.append(inputs[i-12:i, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    predictions_scaled = model.predict(X_test, verbose=0)
    predictions = scaler.inverse_transform(predictions_scaled)
    return pd.Series(predictions.flatten(), index=test_data.index)

def forecast_sarima():
    model_fit = train_sarima_model(train_data)
    return model_fit.predict(start=test_data.index[0], end=test_data.index[-1])

def forecast_arima():
    model_fit = train_arima_model(train_data)
    return model_fit.predict(start=test_data.index[0], end=test_data.index[-1])

def forecast_hw():
    model_fit = train_hw_model(train_data)
    return model_fit.forecast(len(test_data))

def forecast_prophet_lstm():
    prophet_model, lstm_model, scaler = train_prophet_lstm_model(train_data)
    future_df = pd.DataFrame({'ds': test_data.index})
    prophet_forecast = prophet_model.predict(future_df)['yhat'].values
    
    residuals = train_data.values - prophet_model.predict(train_data.reset_index().rename(columns={'Datetime': 'ds', 'PJME_MW': 'y'}))['yhat'].values
    last_12_residuals = residuals[-12:].reshape(-1, 1)
    last_12_residuals_scaled = scaler.transform(last_12_residuals)
    
    X_pred = np.array([last_12_residuals_scaled.flatten()]).reshape(1, 12, 1)
    residual_forecast_scaled = lstm_model.predict(X_pred, verbose=0)
    residual_forecast = scaler.inverse_transform(residual_forecast_scaled).flatten()
    
    final_forecast = prophet_forecast + residual_forecast[0]
    return pd.Series(final_forecast, index=test_data.index)

def forecast_tuned_hybrid():
    arima_result, gru_model, scaler, trend, look_back = train_tuned_arima_gru_model(train_data)
    resid_forecast = arima_result.forecast(steps=len(test_data))
    seasonal_train = STL(train_data, seasonal=13).fit().seasonal
    last_12_months = seasonal_train[-look_back:].values.reshape(-1, 1)
    last_12_months_scaled = scaler.transform(last_12_months)
    X_pred = np.array([last_12_months_scaled.flatten()]).reshape(1, look_back, 1)
    seasonal_forecast_scaled = gru_model.predict(X_pred, verbose=0)
    seasonal_forecast = scaler.inverse_transform(seasonal_forecast_scaled).flatten()
    trend_forecast = trend.iloc[-1]
    hybrid_forecast_values = trend_forecast + seasonal_forecast[0] + resid_forecast.values
    return pd.Series(hybrid_forecast_values, index=test_data.index)

# --- Page Implementations ---
if page == "📈 SARIMA":
    render_model_page("📈 SARIMA Model", lottie_sarima, "SARIMA", "#FF4B4B", forecast_sarima, "The SARIMA model effectively captures the seasonal peaks and troughs, indicating it has learned the yearly pattern well.")

if page == "🧠 LSTM":
    render_model_page("🧠 LSTM Model", lottie_lstm, "LSTM", "#FFC300", lambda: forecast_deep_learning(train_lstm_model, train_data, test_data), "LSTM networks capture the seasonal nature of the data, though with slightly less accuracy on the peaks compared to SARIMA in this instance.")

if page == "📊 ARIMA":
    render_model_page("📊 ARIMA Model", lottie_arima, "ARIMA", "#33FF57", forecast_arima, "The standard ARIMA model captures the general trend but fails to replicate the distinct seasonal peaks, highlighting the importance of a seasonal model for this dataset.")

if page == "❄️ Holt-Winters":
    render_model_page("❄️ Holt-Winters Model", lottie_hw, "Holt-Winters", "#33C4FF", forecast_hw, "The Holt-Winters method is designed to handle both trend and seasonality. The forecast is very competitive, capturing the seasonal pattern with high accuracy.")

if page == "🧠 GRU":
    render_model_page("🧠 GRU Model", lottie_gru, "GRU", "#DA70D6", lambda: forecast_deep_learning(train_gru_model, train_data, test_data), "The GRU model, a variant of LSTM, also shows a strong ability to capture the seasonal patterns in the data, making it a powerful tool for time series forecasting.")

if page == "🤖 Simple RNN":
    render_model_page("🤖 Simple RNN Model", lottie_rnn, "Simple RNN", "#FF6347", lambda: forecast_deep_learning(train_rnn_model, train_data, test_data), "The Simple RNN captures the general seasonality but is less accurate than more complex models like LSTM or GRU, particularly struggling with the magnitude of the peaks and troughs.")

if page == "🔧 Prophet-LSTM Hybrid":
    render_model_page("🔧 Prophet-LSTM Hybrid", lottie_prophet, "Prophet-LSTM", "#20B2AA", forecast_prophet_lstm, "This hybrid model uses Prophet to forecast the main trend and seasonality, while an LSTM models the remaining errors (residuals). This approach leverages Prophet's robust decomposition with LSTM's ability to learn complex patterns in the noise.")

if page == "🔧 Tuned ARIMA-GRU":
    render_model_page("🔧 Tuned ARIMA-GRU", lottie_hybrid, "Tuned ARIMA-GRU", "#4682B4", forecast_tuned_hybrid, "This tuned hybrid model uses `auto_arima` to find the best parameters for the residual component, aiming to improve upon the standard hybrid approach. The forecast appears somewhat muted in its seasonality compared to the historical data.")

if page == "🧩 Combination Model":
    def forecast_combo():
        sarima_forecast = forecast_sarima()
        gru_forecast = forecast_deep_learning(train_gru_model, train_data, test_data)
        lstm_forecast = forecast_deep_learning(train_lstm_model, train_data, test_data)
        rnn_forecast = forecast_deep_learning(train_rnn_model, train_data, test_data)
        arima_forecast = forecast_arima()
        hw_forecast = forecast_hw()
        return (sarima_forecast + gru_forecast + lstm_forecast + rnn_forecast + arima_forecast + hw_forecast) / 6
    render_model_page("🧩 Combination Model", lottie_combo, "Combination (All 6 Models)", "#FFA500", forecast_combo, "The Combination Model averages the forecasts from all six individual models. This ensemble technique often leads to a more robust forecast by canceling out individual model errors.")

if page == "🏆 Model Comparison":
    st.title("🏆 Model Performance Comparison")
    st.markdown("A quantitative comparison of all models based on their performance on the test data.")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        if lottie_compare: st_lottie(lottie_compare, height=200, key="compare")
        st.markdown("- **RMSE:** Root Mean Squared Error.\n- **MAE:** Mean Absolute Error.\n- **MAPE:** Mean Absolute Percentage Error.\n\nFor all metrics, **lower values are better**.")

    with col2:
        st.markdown('<div class="custom-container">', unsafe_allow_html=True)
        st.subheader("Performance Metrics Table")
        
        @st.cache_data
        def calculate_all_metrics():
            metrics = {}
            def get_metrics(name, pred):
                metrics[name] = {'RMSE': np.sqrt(mean_squared_error(test_data, pred)), 'MAE': mean_absolute_error(test_data, pred), 'MAPE (%)': mean_absolute_percentage_error(test_data, pred)}

            with st.spinner("Calculating metrics for all models... This might take a moment."):
                sarima_pred = forecast_sarima()
                get_metrics('SARIMA', sarima_pred)
                lstm_pred = forecast_deep_learning(train_lstm_model, train_data, test_data)
                get_metrics('LSTM', lstm_pred)
                arima_pred = forecast_arima()
                get_metrics('ARIMA', arima_pred)
                hw_pred = forecast_hw()
                get_metrics('Holt-Winters', hw_pred)
                gru_pred = forecast_deep_learning(train_gru_model, train_data, test_data)
                get_metrics('GRU', gru_pred)
                rnn_pred = forecast_deep_learning(train_rnn_model, train_data, test_data)
                get_metrics('Simple RNN', rnn_pred)
                get_metrics('Prophet-LSTM Hybrid', forecast_prophet_lstm())
                get_metrics('Tuned ARIMA-GRU', forecast_tuned_hybrid())
                combo_pred = (sarima_pred + lstm_pred + arima_pred + hw_pred + gru_pred + rnn_pred) / 6
                get_metrics('Combination Model', combo_pred)
            
            return pd.DataFrame(metrics).T

        metrics_df = calculate_all_metrics()
        st.dataframe(metrics_df.style.highlight_min(axis=0, color='rgba(0, 169, 255, 0.3)').format('{:.2f}'))
        st.markdown('</div>', unsafe_allow_html=True)

if page == "🔮 Future Forecasts":
    st.title("🔮 Future Forecasts Table")
    st.markdown("This table shows the predicted power consumption (in MW) for the next 12 months, according to each model trained on the full historical dataset.")

    col1, col2 = st.columns([1, 3])
    with col1:
        if lottie_future: st_lottie(lottie_future, height=250, key="future")

    with col2:
        st.markdown('<div class="custom-container">', unsafe_allow_html=True)
        
        @st.cache_data
        def generate_future_forecasts():
            forecast_df = pd.DataFrame()
            future_dates = pd.date_range(start=df_monthly.index[-1] + pd.DateOffset(months=1), periods=12, freq='M')
            forecast_df['Date'] = future_dates

            with st.spinner("Generating future forecasts for all models... This will take some time."):
                def get_deep_learning_future(model_func):
                    model, scaler = model_func(df_monthly)
                    full_scaled = scaler.transform(df_monthly.values.reshape(-1, 1))
                    inputs = full_scaled[-12:]
                    preds = []
                    current_batch = inputs.reshape((1, 12, 1))
                    for i in range(12):
                        current_pred = model.predict(current_batch, verbose=0)[0]
                        preds.append(current_pred)
                        current_batch = np.append(current_batch[:,1:,:], [[current_pred]], axis=1)
                    future = scaler.inverse_transform(preds)
                    return future.flatten()
                
                sarima_full_fit = SARIMAX(df_monthly, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)).fit(disp=False)
                forecast_df['SARIMA'] = sarima_full_fit.forecast(12).values
                
                hw_full_fit = ExponentialSmoothing(df_monthly, trend='add', seasonal='add', seasonal_periods=12).fit()
                forecast_df['Holt-Winters'] = hw_full_fit.forecast(12).values

                forecast_df['GRU'] = get_deep_learning_future(train_gru_model)
                forecast_df['LSTM'] = get_deep_learning_future(train_lstm_model)
                forecast_df['Simple RNN'] = get_deep_learning_future(train_rnn_model)
                
                arima_full_fit = ARIMA(df_monthly, order=(2, 1, 2)).fit()
                forecast_df['ARIMA'] = arima_full_fit.forecast(12).values

                forecast_df['Combination Model'] = forecast_df[['SARIMA', 'Holt-Winters', 'GRU', 'LSTM', 'Simple RNN', 'ARIMA']].mean(axis=1)

            return forecast_df.set_index('Date')

        future_df = generate_future_forecasts()
        st.dataframe(future_df.style.format('{:.2f}'))
        st.markdown('</div>', unsafe_allow_html=True)


if page == "📜 Conclusion":
    st.title("📜 Final Conclusion and Model Insights")
    st.markdown('<div class="custom-container">', unsafe_allow_html=True)
    st.subheader("Final Interpretation")

    tab1, tab2, tab3 = st.tabs(["Performance on Test Set", "Analysis of Forecasts", "Conclusion on Hybrid Models"])

    with tab1:
        st.markdown("""
        - **The Combination Model** clearly has the lowest RMSE, MAE, and MAPE values by a significant margin. This indicates that a simple linear combination of the individual models' predictions on the test set resulted in the most accurate performance on that historical period.
        - Among the individual models, **GRU** performed the best with the lowest RMSE, MAE, and MAPE.
        - **SARIMA and LSTM** also performed relatively well compared to ARIMA, Holt-Winters, and Simple RNN.
        - The **ARIMA-GRU Hybrid** model, in this specific implementation, had the highest error metrics.
        - The **Prophet-LSTM Hybrid** performed better than the standalone ARIMA, Holt-Winters, and RNN models, and the ARIMA-GRU Hybrid, but not as well as the GRU or the simple Combination Model.
        """)

    with tab2:
        st.markdown("""
        - Visually inspecting the future forecasts, the **Combination Forecast** shows some sharp and potentially unrealistic fluctuations. While it performed best on historical metrics, its future forecast might be overly sensitive.
        - Individual models like **SARIMA, LSTM, GRU, and Holt-Winters** produce forecasts that visually appear to follow the expected seasonal pattern more smoothly.
        - The **Prophet-LSTM Hybrid** also appears to capture the seasonality reasonably well in its future forecast.
        - **ARIMA and Simple RNN** forecasts seem to capture the general trend but might not perfectly hit the seasonal peaks and troughs.
        - The **Tuned ARIMA-GRU Hybrid** forecast exhibits a muted seasonality compared to the historical data.
        """)

    with tab3:
        st.subheader("Final Conclusion on the Tuned ARIMA-GRU Hybrid Approach")
        st.markdown("""
        - The tuned ARIMA-GRU hybrid approach is a valid and conceptually sound method. In this specific implementation, it did show improvement over its untuned version. However, based on the historical test set performance, it did not achieve the same level of accuracy as the simple Combination Model or even some of the standalone models like GRU and Holt-Winters.
        - The visual inspection of its future forecast also suggests it might not fully capture the amplitude of the seasonal swings as well as some other models.
        - Therefore, while the approach has merit, its effectiveness is highly dependent on the specific decomposition method, the chosen orders for the ARIMA models, and the architecture and training of the GRU network. In this instance, other models or simpler combination strategies appear to have yielded better results.
        """)
    
    st.markdown('</div>', unsafe_allow_html=True)