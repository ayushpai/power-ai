import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import openai
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler


import singlestoredb as s2


class DataProcessor:
    def __init__(self, data):
        self.data = data
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.encoder = LabelEncoder()

    def prepare_data(self):
        self.data["weekday_encoded"] = self.encoder.fit_transform(self.data["weekday"])
        self.data = self.data.drop(["date", "weekday"], axis=1)
        X = self.data.drop("demand", axis=1)
        y = self.data["demand"]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )


class PowerDemandModel:
    def __init__(self, data_processor):
        self.data_processor = data_processor
        self.model = LinearRegression()

    def train(self):
        self.model.fit(self.data_processor.X_train, self.data_processor.y_train)

    def evaluate(self):
        y_pred = self.model.predict(self.data_processor.X_test)
        mse = mean_squared_error(self.data_processor.y_test, y_pred)
        r2 = r2_score(self.data_processor.y_test, y_pred)
        return mse, r2, y_pred

    def predict_demand(self, temperature, time_of_day, weekday):
        weekday_encoded = self.data_processor.encoder.transform([weekday])[0]
        input_data = pd.DataFrame(
            {
                "temperature": [temperature],
                "time_of_day": [time_of_day],
                "weekday_encoded": [weekday_encoded],
            }
        )
        return self.model.predict(input_data)[0]


def get_gpt_analysis(temperature, time_of_day, weekday, demand):
    client = openai.OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
    )

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": f"4 Generate a detailed analysis of predicted power demand given the temperature {temperature}°C, time of day {time_of_day}:00, and weekday {weekday}. The model predicted a demand of {demand:.2f} MW.",
            },
            {
                "role": "system",
                "content": "5 SENTENCES MAX GET TO THE POINT NO FLUFF",
            },
        ],
        model="gpt-3.5-turbo",
    )
    return chat_completion.choices[0].message.content


def pricing_model_simulation(
    base_price, high_demand_threshold, high_demand_price, simulated_demand
):
    if simulated_demand > high_demand_threshold:
        total_cost = (high_demand_threshold * base_price) + (
            (simulated_demand - high_demand_threshold) * high_demand_price
        )
    else:
        total_cost = simulated_demand * base_price
    return total_cost


# Load data
@st.cache
def load_data():
    conn = s2.connect(
        "YOUR_SINGLESTORE_DB_URL"
    )
    with conn.cursor() as cur:
        cur.execute("SELECT * FROM power_demand_data")
        data = cur.fetchall()
    df = pd.DataFrame(
        data, columns=["date", "temperature", "time_of_day", "weekday", "demand"]
    )
    print(data)
    return df


data = load_data()
data_processor = DataProcessor(data)
data_processor.prepare_data()
model = PowerDemandModel(data_processor)
model.train()
mse, r2, y_pred = model.evaluate()

# Streamlit pages
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Choose a page", ["Model Evaluation", "Make Prediction", "Pricing Model"]
)

if page == "Model Evaluation":
    st.title("Power Demand Prediction Model Evaluation")
    st.write("Mean Squared Error (MSE):", mse)
    st.write("R-squared:", r2)

    # Plotting the predicted vs actual values
    fig, ax = plt.subplots()
    ax.plot(
        data_processor.y_test.reset_index(drop=True),
        label="Actual Demand",
        color="blue",
    )
    ax.plot(y_pred, label="Predicted Demand", color="red")
    ax.set_title("Actual vs Predicted Power Demand")
    ax.set_xlabel("Test Set Index")
    ax.set_ylabel("Power Demand (MW)")
    ax.legend()
    st.pyplot(fig)

elif page == "Make Prediction":
    st.title("Predict Power Demand")
    temp = st.number_input("Temperature (C)", value=20.0)
    time = st.slider("Time of Day (24hr)", 0, 23, 12)
    weekday = st.selectbox(
        "Weekday",
        options=[
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ],
    )

    if st.button("Predict Demand"):
        prediction = model.predict_demand(temp, time, weekday)
        analysis = get_gpt_analysis(temp, time, weekday, prediction)
        st.write(f"Predicted Power Demand: {prediction:.2f} MW")
        st.write("AI Analysis:")
        st.write(analysis)

elif page == "Pricing Model":
    st.title("Simulate Pricing Models Based on Power Demand")

    # Inputs for pricing thresholds
    base_price = st.number_input(
        "Base Price per MW ($)", min_value=0.0, value=50.0, step=1.0
    )
    high_demand_threshold = st.number_input(
        "High Demand Threshold (MW)", min_value=1000, value=1500, step=50
    )
    high_demand_price = st.number_input(
        "Price per MW above high demand threshold ($)",
        min_value=0.0,
        value=75.0,
        step=1.0,
    )

    # Interactive sliders for adjusting prediction for simulation
    st.write("Adjust the following to simulate demand and see corresponding pricing:")
    simulated_demand = st.slider(
        "Simulated Power Demand (MW)",
        min_value=500,
        max_value=3000,
        value=1500,
        step=50,
    )

    # Calculate cost based on pricing model
    if simulated_demand > high_demand_threshold:
        total_cost = (high_demand_threshold * base_price) + (
            (simulated_demand - high_demand_threshold) * high_demand_price
        )
    else:
        total_cost = simulated_demand * base_price

    st.write(f"Total Cost for {simulated_demand} MW: ${total_cost:,.2f}")

    # Explanation text
    st.markdown(
        """
    This page allows you to simulate different pricing models based on power demand. 
    Adjust the base price and the high demand threshold to see how different scenarios affect the total cost.
    """
    )

    # Optionally, plot the cost as a function of demand
    demands = list(range(500, 3001, 50))
    costs = [
        (
            (d * base_price)
            if d <= high_demand_threshold
            else (high_demand_threshold * base_price)
            + ((d - high_demand_threshold) * high_demand_price)
        )
        for d in demands
    ]
    fig, ax = plt.subplots()
    ax.plot(demands, costs, label="Total Cost")
    ax.set_title("Cost vs. Demand")
    ax.set_xlabel("Demand (MW)")
    ax.set_ylabel("Total Cost ($)")
    ax.grid(True)
    st.pyplot(fig)
