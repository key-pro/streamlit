import pandas as pd
from utils.data_fetcher import fetch_fx_data
from components.charts import create_candlestick_chart
import os

df = fetch_fx_data("USD/JPY", "1mo", "1d")
fig = create_candlestick_chart(df, 'USD/JPY')

with open("test_chart.html", "w") as f:
    f.write(fig.to_html(full_html=True, include_plotlyjs='cdn'))
print("Saved test_chart.html")
