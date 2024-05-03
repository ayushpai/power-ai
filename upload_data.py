import singlestoredb as s2
import os
import pandas as pd

conn = s2.connect(
    "ayush:Test1234@svc-3482219c-a389-4079-b18b-d50662524e8a-shared-dml.aws-virginia-6.svc.singlestore.com:3333/database_79fb0"
)


# create a table in singlestore with the data from Power_Demand_Data.csv


def create_table():
    with conn.cursor() as cur:
        cur.execute(
            "CREATE TABLE power_demand_data (date DATE, temperature FLOAT, time_of_day FLOAT, weekday VARCHAR(255), demand FLOAT)"
        )
        conn.commit()

    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv("Power_Demand_Data.csv")

    # Insert the data into the database
    for index, row in df.iterrows():
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO power_demand_data (date, temperature, time_of_day, weekday, demand) VALUES (%s, %s, %s, %s, %s)",
                (
                    row["date"],
                    row["temperature"],
                    row["time_of_day"],
                    row["weekday"],
                    row["demand"],
                ),
            )
            conn.commit()


# pull the data from the database and put in a pandas DataFrame
def get_data():
    with conn.cursor() as cur:
        cur.execute("SELECT * FROM power_demand_data")
        data = cur.fetchall()
    df = pd.DataFrame(
        data, columns=["date", "temperature", "time_of_day", "weekday", "demand"]
    )
    return df


# delete tables from the database
def delete_table():
    with conn.cursor() as cur:
        cur.execute("DROP TABLE power_demand_data")
        conn.commit()


print(get_data())
