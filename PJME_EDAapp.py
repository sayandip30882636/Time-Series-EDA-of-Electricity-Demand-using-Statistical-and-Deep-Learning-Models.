# app.py
import streamlit as st
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date
import holidays
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
import seaborn as sns
import streamlit.components.v1 as components

# --- Page Configuration and Professional Styling ---
st.set_page_config(
    page_title="PJME Power Consumption EDA",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Advanced CSS for a cinematic, attractive look with a blue theme
st.markdown("""
<style>
    /* Main app background image with a blue overlay */
    [data-testid="stAppViewContainer"] > .main {
        background-image: linear-gradient(rgba(0, 15, 40, 0.7), rgba(0, 15, 40, 0.7)), url("https://images.alphacoders.com/133/1330023.png");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: rgba(10, 20, 35, 0.8);
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(0, 169, 255, 0.3);
    }
    /* Headers styling with text shadow for readability */
    h1, h2, h3 {
        font-weight: 700;
        color: #FFFFFF;
        text-shadow: 2px 2px 6px rgba(0, 0, 0, 0.9);
    }
    /* "Glassmorphism" containers */
    .custom-container {
        border-radius: 0.75rem;
        padding: 1.5rem;
        background: rgba(20, 30, 50, 0.75);
        backdrop-filter: blur(15px);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        margin-bottom: 2rem;
        border: 1px solid rgba(0, 169, 255, 0.3);
    }
    /* Make text inside containers readable */
    .st-emotion-cache-16txtl3, .st-emotion-cache-16txtl3 *, .st-emotion-cache-16txtl3 h3 {
        color: #FFFFFF !important;
        text-shadow: none;
    }
    /* Improved Navigation Bar Styling */
    div[data-baseweb="radio"] > div > label {
        background-color: transparent;
        padding: 12px 18px;
        margin-bottom: 8px;
        border-radius: 0.5rem;
        transition: all 0.3s ease;
        border: 1px solid rgba(0, 169, 255, 0.2);
        width: 100%;
        font-weight: 500;
        color: #E0E0E0;
    }
    div[data-baseweb="radio"] > div > label:hover {
        background-color: rgba(0, 169, 255, 0.2);
        color: #FFFFFF;
        border: 1px solid #00A9FF;
        transform: scale(1.02);
    }
    /* Custom styling for alerts */
    [data-testid="stAlert"] {
        background: rgba(0, 169, 255, 0.15);
        border: 1px solid rgba(0, 169, 255, 0.3);
        border-radius: 0.5rem;
        color: #FFFFFF;
    }
    [data-testid="stAlert"] svg {
        fill: #00A9FF;
    }
    div[data-baseweb="radio"] input[type="radio"] {
        display: none;
    }
</style>
""", unsafe_allow_html=True)

# --- Functions from the Notebook (Unaltered Logic) ---

@st.cache_data
def load_and_process_data(file_path='PJME_hourly.csv'):
    df = pd.read_csv(file_path)
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df = df.set_index('Datetime')
    df = df.rename(columns={'PJME_MW': 'value'})
    if not df.index.is_monotonic_increasing:
        df = df.sort_index()
    return df

def plot_dataset_st(df, title, y_label="Value"):
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_alpha(0.0)
    ax.set_facecolor((0, 0, 0, 0))
    ax.plot(df.index, df.value, color='#00A9FF', linewidth=0.7)
    ax.set_title(title, fontsize=16, weight='bold', color='#FFFFFF')
    ax.set_xlabel("Date", color='#FFFFFF')
    ax.set_ylabel(y_label, color='#FFFFFF')
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('white')
    fig.tight_layout()
    st.pyplot(fig)

def generate_time_lags(df, n_lags):
    df_n = df.copy()
    for n in range(1, n_lags + 1):
        df_n[f"lag{n}"] = df_n["value"].shift(n)
    df_n = df_n.iloc[n_lags:]
    return df_n

def onehot_encode_pd(df, cols):
    df_out = df.copy()
    for col in cols:
        dummies = pd.get_dummies(df_out[col], prefix=col)
        df_out = pd.concat([df_out, dummies], axis=1).drop(columns=[col])
    return df_out

def generate_cyclical_features(df, col_name, period, start_num=0):
    kwargs = {
        f'sin_{col_name}' : lambda x: np.sin(2*np.pi*(df[col_name]-start_num)/period),
        f'cos_{col_name}' : lambda x: np.cos(2*np.pi*(df[col_name]-start_num)/period)
             }
    return df.assign(**kwargs).drop(columns=[col_name])

us_holidays = holidays.US()
def is_holiday(date):
    date = date.replace(hour = 0)
    return 1 if (date in us_holidays) else 0

def add_holiday_col(df, holidays):
    return df.assign(is_holiday = df.index.to_series().apply(is_holiday))

# --- Load Data ---
try:
    df_main = load_and_process_data()
except FileNotFoundError:
    st.error("`PJME_hourly.csv` not found. Please place it in the same directory as the app.")
    st.stop()

# --- Sidebar Navigation ---
st.sidebar.title("📊 EDA Dashboard")
st.sidebar.markdown("Navigate through the analysis sections:")

page = st.sidebar.radio("Navigation", [
    "🏠 Introduction",
    "📈 Data Visualization",
    "🛠️ Feature Engineering",
    "🔍 Decomposition Analysis",
    "📉 Stationarity Analysis",
    "🔗 Correlation Analysis",
    "🎯 Feature-Target Relationships"
], label_visibility="collapsed")
st.sidebar.markdown("---")
st.sidebar.subheader("Tech Stack")
st.sidebar.markdown("""
The UI/UX of this dashboard is crafted with **🎨 Streamlit**, an open-source framework for building beautiful data applications.

- **🐍 Pandas:** Powers all data manipulation and processing.
- **📊 Matplotlib & Seaborn:** Used to render all static and interactive visualizations.
- **📈 Statsmodels:** Drives the core time-series analysis, including decomposition and stationarity tests.
- **📅 holidays:** Provides the US holiday feature data.
- **🧠 PyTorch:** Serves as the underlying machine learning environment.
""")
st.sidebar.markdown("---")


# --- Main Application Pages ---

if page == "🏠 Introduction":
    st.title("⚡ PJME Power Consumption: An Advanced EDA")
    st.markdown("### Welcome to the Exploratory Data Analysis Dashboard")

    st.markdown('<div class="custom-container">', unsafe_allow_html=True)
    st.markdown("""
    This advanced interactive application presents a detailed analysis of the PJM East (PJME) region's hourly electricity consumption. The goal is to uncover trends, seasonal patterns, and other insights that are crucial for building accurate forecasting models. All analyses and interpretations are sourced directly from the provided analysis notebook and presented in a professional, intuitive UI.
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="custom-container">', unsafe_allow_html=True)
    st.subheader("Dataset at a Glance")
    
    # --- Prepare values for animation ---
    start_date = df_main.index.min().strftime('%Y-%m-%d')
    end_date = df_main.index.max().strftime('%Y-%m-%d')
    total_obs = len(df_main)
    min_power = df_main['value'].min()
    max_power = df_main['value'].max()
    avg_power = df_main['value'].mean()

    # --- HTML and JavaScript for Animated Metrics ---
    components.html(f"""
    <script src="https://unpkg.com/countup.js@2.0.7/dist/countUp.umd.js"></script>
    <style>
        .metric-container {{
            display: flex;
            flex-direction: column;
            padding: 1rem;
            border-radius: 0.5rem;
            background-color: rgba(0, 0, 0, 0.2);
            border: 1px solid rgba(0, 169, 255, 0.2);
        }}
        .metric-label {{
            font-size: 0.9rem;
            color: #E0E0E0;
        }}
        .metric-value {{
            font-size: 1.75rem;
            font-weight: 600;
            color: #FFFFFF;
        }}
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 1rem;
            margin-bottom: 1rem;
        }}
    </style>
    
    <div class="metric-grid">
        <div class="metric-container">
            <div class="metric-label">Start Date</div>
            <div class="metric-value">{start_date}</div>
        </div>
        <div class="metric-container">
            <div class="metric-label">End Date</div>
            <div class="metric-value">{end_date}</div>
        </div>
        <div class="metric-container">
            <div class="metric-label">Total Observations</div>
            <div class="metric-value" id="totalObs">0</div>
        </div>
    </div>
    
    <hr style="border-color: rgba(0, 169, 255, 0.2);">
    <p style="font-weight: 500; margin-top: 1.5rem; color: #FFFFFF;">Power Consumption (MW)</p>
    <div class="metric-grid">
        <div class="metric-container">
            <div class="metric-label">Minimum Consumption</div>
            <div class="metric-value" id="minPower">0</div>
        </div>
        <div class="metric-container">
            <div class="metric-label">Maximum Consumption</div>
            <div class="metric-value" id="maxPower">0</div>
        </div>
        <div class="metric-container">
            <div class="metric-label">Average Consumption</div>
            <div class="metric-value" id="avgPower">0</div>
        </div>
    </div>

    <script>
        const options = {{
            duration: 2.5, // Animation duration in seconds
            useGrouping: true,
        }};
        
        const totalObs = new countUp.CountUp('totalObs', {total_obs}, options);
        const minPower = new countUp.CountUp('minPower', {min_power}, options);
        const maxPower = new countUp.CountUp('maxPower', {max_power}, options);
        const avgPower = new countUp.CountUp('avgPower', {avg_power}, {{...options, decimalPlaces: 0}});

        if (!totalObs.error && !minPower.error && !maxPower.error && !avgPower.error) {{
            totalObs.start();
            minPower.start();
            maxPower.start();
            avgPower.start();
        }} else {{
            console.error(totalObs.error);
        }}
    </script>
    """, height=350) 
    
    st.dataframe(df_main.head())
    st.markdown('</div>', unsafe_allow_html=True)

elif page == "📈 Data Visualization":
    st.title("📈 Data Visualization")
    st.markdown("Visualizing the data at different time resolutions helps smooth out noise and reveal underlying trends and seasonality more clearly.")
    st.markdown('<div class="custom-container">', unsafe_allow_html=True)
    option = st.selectbox("Select Time Resolution", ('Hourly', 'Daily', 'Weekly', 'Monthly', 'Yearly'))
    if option == 'Hourly':
        plot_dataset_st(df_main, title='Hourly Energy Consumption')
    else:
        resampled_df = df_main.resample(option[0]).mean()
        plot_dataset_st(resampled_df, title=f'{option} Average Energy Consumption')
    st.info("""
    **Interpretation:**
    - **Hourly:** Very noisy, making it hard to see clear trends.
    - **Daily:** Smoother than hourly, revealing daily fluctuations and longer-term trends.
    - **Weekly/Monthly:** Clearly show seasonal patterns (higher demand in summer/winter).
    - **Yearly:** Highlights the long-term growth or decline in overall consumption.
    """, icon="💡")
    st.markdown('</div>', unsafe_allow_html=True)

elif page == "🛠️ Feature Engineering":
    st.title("🛠️ Feature Engineering")
    st.markdown("We create new features from the datetime index to provide explicit time-based context to the model.")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Time Lags", "Date/Time Predictors", "Cyclical Features", "Holiday Feature", "One-Hot Encoding"])

    with tab1:
        st.subheader("Generating Time-Lagged Observations")
        input_dim = st.slider("Select number of lags", 10, 200, 100)
        df_timelags = generate_time_lags(df_main, input_dim)
        st.dataframe(df_timelags.head())
        st.info(f"""
        **Interpretation:** This creates a 'sliding window' of the past {input_dim} hours. Each `lagN` column contains the energy demand from N hours ago. This is crucial for models to learn the relationship between current demand and the demand in the recent past.
        """, icon="💡")
    with tab2:
        st.subheader("Generating Date/Time Predictors")
        df_features = df_main.assign(hour=df_main.index.hour, day=df_main.index.day, month=df_main.index.month, day_of_week=df_main.index.dayofweek, week_of_year=df_main.index.isocalendar().week)
        st.dataframe(df_features.head())
        st.info("""
        **Interpretation:** These are the foundational time-based features extracted directly from the index. They provide raw numerical representations of time components like the hour of the day or the day of the week. These raw features serve as the basis for more advanced transformations like one-hot encoding or cyclical feature generation.
        """, icon="💡")
    with tab3:
        st.subheader("Generating Cyclical Features (Sine/Cosine)")
        df_features_cyclical = generate_cyclical_features(df_features, 'hour', 24, 0)
        st.dataframe(df_features_cyclical[['sin_hour', 'cos_hour']].head())
        st.info("""
        **Interpretation:** Cyclical features like 'hour' are poorly represented as simple numbers (23 is not 'far' from 0). Sine and cosine transformations map the hour onto a circle, preserving this cyclical relationship and allowing the model to understand that midnight is adjacent to 11 PM.
        """, icon="💡")
    with tab4:
        st.subheader("Generating Holiday Feature")
        df_features_holiday = add_holiday_col(df_main, us_holidays)
        st.dataframe(df_features_holiday[df_features_holiday['is_holiday'] == 1].head())
        st.info("""
        **Interpretation:** We add a binary `is_holiday` feature to explicitly flag US public holidays. This is important as energy consumption patterns are often significantly different on these days due to changes in business and social activities.
        """, icon="💡")
    with tab5:
        st.subheader("One-Hot Encoding")
        df_features_ohe = onehot_encode_pd(df_features, ['month', 'day_of_week'])
        st.dataframe(df_features_ohe.head())
        st.info("""
        **Interpretation:** One-hot encoding converts categorical features (like month) into a numerical format. It creates a binary column for each category, preventing the model from assuming a false linear relationship (e.g., that December is 'larger' than January).
        """, icon="💡")

elif page == "🔍 Decomposition Analysis":
    st.title("🔍 Time Series Decomposition")
    st.markdown("Decomposition separates the time series into its constituent components: **Trend**, **Seasonality**, and **Residuals** (noise). This helps in understanding the underlying structure of the data.")
    st.markdown('<div class="custom-container">', unsafe_allow_html=True)
    decomp_option = st.selectbox("Select Time Resolution for Decomposition", ('Daily', 'Monthly', 'Yearly'))
    if decomp_option == 'Daily':
        decomposition = seasonal_decompose(df_main.resample('D').mean().dropna()['value'], model='additive', period=365)
        st.pyplot(decomposition.plot())
    elif decomp_option == 'Monthly':
        decomposition = seasonal_decompose(df_main.resample('M').mean().dropna()['value'], model='additive', period=12)
        st.pyplot(decomposition.plot())
    elif decomp_option == 'Yearly':
        decomposition = seasonal_decompose(df_main.resample('Y').mean().dropna()['value'], model='additive', period=1)
        st.pyplot(decomposition.plot())
    st.info("""
    **Interpretation:** The plots reveal the long-term trend (e.g., overall increase/decrease in consumption), the repeating seasonal patterns (e.g., yearly highs and lows), and the random, unpredictable noise left over in the data.
    """, icon="💡")
    st.markdown('</div>', unsafe_allow_html=True)

elif page == "📉 Stationarity Analysis":
    st.title("📉 Stationarity Analysis")
    st.markdown("Stationarity means the statistical properties of a series (mean, variance) are constant over time. We use the **Augmented Dickey-Fuller (ADF) test** to check this. A **p-value <= 0.05** indicates the data is likely stationary.")
    stationarity_option = st.selectbox("Select Data for Stationarity Analysis", ('Daily', 'Weekly', 'Monthly'))

    def display_stationarity_analysis(series, name):
        adf_test = adfuller(series.dropna())
        st.markdown(f'<div class="custom-container"><h3>Analysis for {name} Data</h3>', unsafe_allow_html=True)
        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric("ADF Statistic", f"{adf_test[0]:.4f}")
            st.metric("p-value", f"{adf_test[1]:.4g}")
            if adf_test[1] <= 0.05:
                st.success("The data is likely **stationary**.")
            else:
                st.warning("The data is likely **non-stationary**.")
        with col2:
            fig_acf, ax_acf = plt.subplots(figsize=(8,4))
            plot_acf(series.dropna(), ax=ax_acf, lags=50)
            ax_acf.set_title(f'ACF Plot for {name} Data')
            st.pyplot(fig_acf)
        st.markdown('</div>', unsafe_allow_html=True)

    if stationarity_option == 'Daily':
        display_stationarity_analysis(df_main.resample('D').mean()['value'], 'Daily')
    elif stationarity_option == 'Weekly':
        display_stationarity_analysis(df_main.resample('W').mean()['value'], 'Weekly')
    elif stationarity_option == 'Monthly':
        monthly_series = df_main.resample('M').mean()['value']
        display_stationarity_analysis(monthly_series, 'Original Monthly')
        st.info("Since the original monthly data is non-stationary, we apply first-order differencing to stabilize it.", icon="ℹ️")
        display_stationarity_analysis(monthly_series.diff(), 'Differenced Monthly')


elif page == "🔗 Correlation Analysis":
    st.title("🔗 Correlation Analysis")
    st.markdown("Here we examine the statistical relationships between variables and the target.")
    st.markdown('<div class="custom-container">', unsafe_allow_html=True)
    st.subheader("Time Lagged Correlation Analysis")
    df_timelags_corr = generate_time_lags(df_main, 100)
    correlations = df_timelags_corr.corr()
    fig_lag, ax_lag = plt.subplots(figsize=(12, 6))
    ax_lag.plot(range(1, 101), correlations['value'].iloc[1:], marker='o', linestyle='-', markersize=4, color='#1E3A5F')
    ax_lag.set_title('Time-Lagged Correlations of PJME_MW')
    ax_lag.axhline(0, color='red', linestyle='--')
    ax_lag.grid(True, linestyle='--')
    st.pyplot(fig_lag)
    st.info("""
    **Interpretation:** This plot shows the correlation of the current demand with demand from previous hours. The high positive correlation at small lags confirms a strong short-term dependency. The decaying pattern shows that the influence of past values diminishes over time.
    """, icon="💡")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="custom-container">', unsafe_allow_html=True)
    st.subheader("Correlation Matrix of Features")
    df_features_corr = df_main.assign(hour=df_main.index.hour, day_of_week=df_main.index.dayofweek, month=df_main.index.month, is_holiday=df_main.index.to_series().apply(is_holiday))
    df_features_corr = generate_cyclical_features(df_features_corr, 'hour', 24, 0)
    fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
    sns.heatmap(df_features_corr.corr(), annot=True, cmap='coolwarm', ax=ax_corr, fmt=".2f")
    st.pyplot(fig_corr)
    st.info("""
    **Interpretation:** The heatmap quantifies the linear relationships between engineered features and the target (`value`). We can see moderate correlations for the cyclical hour features (`sin_hour`, `cos_hour`) and day of the week (`day_of_week`), confirming their relevance for prediction.
    """, icon="💡")
    st.markdown('</div>', unsafe_allow_html=True)

elif page == "🎯 Feature-Target Relationships":
    st.title("🎯 Feature-Target Relationships")
    st.markdown("These plots reveal key patterns by showing the average energy demand grouped by different time-based features.")
    st.markdown('<div class="custom-container">', unsafe_allow_html=True)
    df_features_viz = df_main.assign(hour=df_main.index.hour, day_of_week=df_main.index.dayofweek, month=df_main.index.month, is_holiday=df_main.index.to_series().apply(is_holiday))
    viz_option = st.selectbox("Select Feature to Visualize", ('Hour', 'Day of Week', 'Month', 'Holiday Status'))
    fig_viz, ax_viz = plt.subplots(figsize=(12, 6))
    if viz_option == 'Hour':
        df_features_viz.groupby('hour')['value'].mean().plot(ax=ax_viz, marker='o')
        ax_viz.set_title('Average Consumption by Hour')
    elif viz_option == 'Day of Week':
        df_features_viz.groupby('day_of_week')['value'].mean().plot(ax=ax_viz, marker='o')
        ax_viz.set_title('Average Consumption by Day of Week')
        ax_viz.set_xticklabels(['', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    elif viz_option == 'Month':
        df_features_viz.groupby('month')['value'].mean().plot(ax=ax_viz, marker='o')
        ax_viz.set_title('Average Consumption by Month')
    elif viz_option == 'Holiday Status':
        df_features_viz.groupby('is_holiday')['value'].mean().plot(kind='bar', ax=ax_viz)
        ax_viz.set_title('Average Consumption by Holiday Status')
        ax_viz.set_xticklabels(['Non-Holiday', 'Holiday'], rotation=0)
    ax_viz.set_ylabel('Average Consumption (MW)')
    ax_viz.grid(True, linestyle='--')
    st.pyplot(fig_viz)
    st.info("""
    **Interpretation:** These plots visually confirm the presence of strong daily, weekly, and yearly seasonalities, as well as the impact of holidays. For example, the 'Hour' plot shows the typical drop in demand overnight, while the 'Month' plot shows peaks in summer and winter.
    """, icon="💡")
    st.markdown('</div>', unsafe_allow_html=True)