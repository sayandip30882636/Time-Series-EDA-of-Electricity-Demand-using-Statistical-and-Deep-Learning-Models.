# Hybrid-Time-Series-Forecasting-of-Electricity-Demand-using-Statistical-and-Deep-Learning-Models.

# ⚡ PJME Power Consumption: Forecasting Dashboard

# For EDA " Click Below "

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://hybrid-time-series-forecasting.streamlit.app/)

# For Analytical "Click Below "



An interactive Streamlit dashboard for visualizing, comparing, and forecasting the monthly power consumption of the PJM East (PJME) region. This application offers a hands-on interface for comprehensive time-series analysis, utilizing multiple statistical and deep learning models to predict future energy demand.

## 🔮 Features

This dashboard transforms a complex Jupyter Notebook analysis into a user-friendly and cinematic web application with the following features:

- **Cinematic UI:** a beautiful user interface with an anime-inspired background and modern "glassmorphism" containers.
- **Interactive Visualizations:** All plots are rendered with a theme that matches the UI, providing a cohesive and professional look.
- **Comprehensive Model Analysis:** Explore a wide range of forecasting models, including:
  - **Statistical Models:** SARIMA, ARIMA, Holt-Winters.
  - **Deep Learning Models:** LSTM, GRU, and Simple RNN.
  - **Advanced Hybrid Models:** Prophet-LSTM and a Tuned ARIMA-GRU.
  - **Ensemble Methods:** A Combination Model that averages the predictions of all individual models.
- **Performance Comparison:** A detailed comparison table that ranks all models based on key performance metrics (RMSE, MAE, MAPE).
- **Future Forecasts:** A dedicated page showing a table of the predicted power consumption for the next 12 months from each model.
- **In-Depth Interpretations:** Each model and analysis step is accompanied by a clear, summarized interpretation taken directly from the original research.

---

## 🛠️ Tech Stack

This project was built using a combination of powerful data science and web development libraries:

- **Application Framework:** Streamlit
- **Data Manipulation:** Pandas
- **Visualizations:** Matplotlib, Seaborn
- **Statistical Modeling:** Statsmodels, Pmdarima, Prophet
- **Deep Learning:** TensorFlow (Keras)
- **UI Enhancements:** Streamlit Lottie for animations

---

## 🚀 Getting Started

To run this application on your local machine, follow these steps.

### Prerequisites

Ensure you have Python 3.8 or later installed on your system.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repository-name.git](https://github.com/your-username/your-repository-name.git)
    cd your-repository-name
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    # For Windows
    python -m venv venv
    venv\Scripts\activate

    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required packages:**
    The `requirements.txt` file contains all the necessary libraries. Install them with a single command:
    ```bash
    pip install -r requirements.txt
    ```

### Running the App

1.  **Place the dataset:** Ensure that the `PJME_hourly.csv` file is in the root directory of the project.

2.  **Launch the Streamlit app:**
    Run the following command in your terminal:
    ```bash
    streamlit run app.py
    ```

Your browser should automatically open a new tab with the running application.

---

## 🚢 Deployment

This application is designed for deployment on **Streamlit Community Cloud**. To deploy your own version:

1.  **Push your project to a public GitHub repository.** Make sure `app.py`, `requirements.txt`, and `PJME_hourly.csv` are in the repository.
2.  **Sign up or log in** to [Streamlit Community Cloud](https://share.streamlit.io/).
3.  Click **"New app"** and connect your GitHub account.
4.  Select your repository, the `main` branch, and `app.py` as the main file path.
5.  Click **"Deploy!"**.

---

## 📜 Conclusion of Analysis

The comprehensive analysis performed in this project concludes that a **Combination Model** (averaging the forecasts of all individual models) provides the most accurate predictions on the historical test set, as measured by RMSE, MAE, and MAPE.

However, for generating future forecasts, individual models like **GRU** and **Holt-Winters** produce more stable and visually plausible seasonal patterns. This highlights the important trade-off between performance on a historical test set and the robustness of future predictions.

