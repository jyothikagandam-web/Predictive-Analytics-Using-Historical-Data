
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Predictive Analytics Dashboard", layout="wide")

st.title("Sales Trend Analysis and Future Prediction")
st.write("Predict future sales using historical data and Machine Learning.")

@st.cache_data
def load_data():
    df = pd.read_csv("train.csv")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["sales"] = pd.to_numeric(df["sales"], errors="coerce")
    df = df.dropna(subset=["date", "sales"])
    return df

df = load_data()

st.subheader("1. Historical Data Overview")

c1, c2, c3 = st.columns(3)
c1.metric("Total Records", f"{len(df):,}")
c2.metric("Total Sales", f"{df['sales'].sum():,.0f}")
c3.metric("Average Daily Sales", f"{df['sales'].mean():,.2f}")

st.write("Dataset Preview")
st.dataframe(df.head(10), use_container_width=True)

st.subheader("2. Data Preprocessing")

monthly_sales = (
    df.set_index("date")["sales"]
    .resample("MS")
    .sum()
    .reset_index()
)

monthly_sales.columns = ["Date", "Sales"]

st.success("Data cleaned and converted into monthly sales records.")

st.write("Monthly Sales Data")
st.dataframe(monthly_sales, use_container_width=True)

st.subheader("3. Historical Sales Trend")

fig1 = px.line(
    monthly_sales,
    x="Date",
    y="Sales",
    title="Monthly Historical Sales"
)
st.plotly_chart(fig1, use_container_width=True)


def create_features(values, index, date):
    return {
        "month": date.month,
        "year": date.year,
        "time_index": index,
        "lag_1": values[-1],
        "lag_2": values[-2],
        "lag_3": values[-3],
        "lag_12": values[-12],
        "rolling_3": np.mean(values[-3:]),
        "rolling_12": np.mean(values[-12:])
    }


sales_values = monthly_sales["Sales"].tolist()
dates = monthly_sales["Date"].tolist()

if len(sales_values) < 24:
    st.error("At least 24 months of sales data are required.")
    st.stop()

features = []
targets = []

for i in range(12, len(sales_values)):
    feature = create_features(
        sales_values[:i],
        i,
        dates[i]
    )
    features.append(feature)
    targets.append(sales_values[i])

X = pd.DataFrame(features)
y = np.array(targets)

st.subheader("4. Model Training and Evaluation")

test_size = 12

X_train = X.iloc[:-test_size]
X_test = X.iloc[-test_size:]
y_train = y[:-test_size]
y_test = y[-test_size:]

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    max_depth=8
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

c1, c2, c3 = st.columns(3)
c1.metric("MAE", f"{mae:,.2f}")
c2.metric("RMSE", f"{rmse:,.2f}")
c3.metric("R² Score", f"{r2:.3f}")

st.write("Model trained using historical records. The last 12 months were reserved for testing.")

comparison = pd.DataFrame({
    "Date": dates[-test_size:],
    "Actual Sales": y_test,
    "Predicted Sales": predictions
})

st.subheader("5. Actual vs Predicted Sales")

comparison_chart = px.line(
    comparison,
    x="Date",
    y=["Actual Sales", "Predicted Sales"],
    title="Actual Sales vs Model Predictions"
)
st.plotly_chart(comparison_chart, use_container_width=True)

st.dataframe(comparison, use_container_width=True)

st.subheader("6. Future Sales Forecast")

final_model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    max_depth=8
)

final_model.fit(X, y)

history = sales_values.copy()
future_dates = []
future_predictions = []

last_date = dates[-1]

for i in range(1, 13):
    next_date = last_date + pd.DateOffset(months=i)

    feature = create_features(
        history,
        len(history),
        next_date
    )

    next_prediction = final_model.predict(
        pd.DataFrame([feature])
    )[0]

    next_prediction = max(0, next_prediction)

    history.append(next_prediction)
    future_dates.append(next_date)
    future_predictions.append(next_prediction)

forecast = pd.DataFrame({
    "Date": future_dates,
    "Predicted Sales": future_predictions
})

c1, c2 = st.columns(2)
c1.metric("Next 12 Months Forecast", f"{sum(future_predictions):,.0f}")
c2.metric("Average Forecasted Monthly Sales", f"{np.mean(future_predictions):,.2f}")

forecast_chart = px.line(
    forecast,
    x="Date",
    y="Predicted Sales",
    markers=True,
    title="Future 12 Months Sales Forecast"
)

st.plotly_chart(forecast_chart, use_container_width=True)

st.dataframe(forecast, use_container_width=True)

st.subheader("7. Download Results")

csv = forecast.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Future Sales Predictions",
    data=csv,
    file_name="sales_forecast.csv",
    mime="text/csv"
)

st.success("Predictive Analytics Dashboard Completed!")
