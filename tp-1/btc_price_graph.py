import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV
df = pd.read_csv("btc_daily_price.csv")

tv = list(range(1, 4650))

# Parse dates
df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")

# Clean Price (remove commas, convert to float)
df["Price"] = df["Price"].str.replace(",", "", regex=False).astype(float)

# Sort by date ascending
df = df.sort_values("Date")

# Convert datetime to year fraction
def to_year_fraction(d):
    year = d.year
    start = pd.Timestamp(year=year, month=1, day=1)
    end   = pd.Timestamp(year=year+1, month=1, day=1)
    return year + (d - start).days / (end - start).days

df["YearFrac"] = df["Date"].apply(to_year_fraction)

# Double log plot (year vs price)
plt.figure(figsize=(11,6))
plt.loglog(tv, df["Price"], linewidth=0.7)

plt.xlabel("Year (log scale)")
plt.ylabel("Bitcoin Price (USD, log scale)")
plt.title("Bitcoin Daily Close — Double Log Plot (Years vs Price)")
plt.grid(True, which="both", ls="--")
plt.tight_layout()
plt.show()
