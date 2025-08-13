import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/processed/housing_processed.csv")

plt.figure(figsize=(8,6))
sns.histplot(df["median_house_value"], bins=50, kde=True)
plt.title("Distribution of House Prices")
plt.savefig("data/processed/price_distribution.png")
plt.show()