from utils.data_fetcher import fetch_fx_data
import os

df = fetch_fx_data("USD/JPY", "1mo", "1d")
print("HEAD:\n", df.head())
print("\nINDEX TYPE:", type(df.index))
print("\nDF INFO:\n", df.info())
