import pandas as pd
from utils.data_fetcher import fetch_fx_data

df = fetch_fx_data("USDJPY=X", "5d", "15m")
print(f"Row count: {len(df)}")
if not df.empty:
    print(df.head(2))
    print(df.tail(2))
