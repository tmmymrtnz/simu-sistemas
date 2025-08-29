import pandas as pd
import matplotlib.pyplot as plt

# Load CSV
df = pd.read_csv("btc_daily_price.csv")

# Parse date & price
df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")
df["Price"] = (df["Price"].astype(str)
               .str.replace(",", "", regex=False)
               .astype(float))

# Ordenar y limpiar
df = df.sort_values("Date").dropna(subset=["Price"])
df = df[df["Price"] > 0]

# Año fraccional (eje X)
def to_year_fraction(d):
    year = d.year
    start = pd.Timestamp(year=year, month=1, day=1)
    end   = pd.Timestamp(year=year+1, month=1, day=1)
    return year + (d - start).days / (end - start).days

df["YearFrac"] = df["Date"].apply(to_year_fraction)

x = df["YearFrac"].values
y = df["Price"].values

# ---- Plots lado a lado ----
fig, axes = plt.subplots(1, 2, figsize=(14,6))

# Escala lineal
axes[0].plot(x, y, linewidth=0.7)
axes[0].set_xlabel("Year")
axes[0].set_ylabel("Bitcoin Price (USD)")
axes[0].set_title("Bitcoin Price (Linear Scale)")
axes[0].grid(True, ls="--")

# Escala log-log
axes[1].loglog(x, y, linewidth=0.7)
axes[1].set_xlabel("Year (log scale)")
axes[1].set_ylabel("Bitcoin Price (log scale)")
axes[1].set_title("Bitcoin Price (Log-Log Scale)")
axes[1].grid(True, which="both", ls="--")

plt.tight_layout()
plt.show()
