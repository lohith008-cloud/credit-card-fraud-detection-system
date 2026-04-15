import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/creditcard.csv")

print(df.info())
print(df.describe())

# Class imbalance
sns.countplot(x="Class", data=df)
plt.title("Class Distribution")
plt.show()

# Amount distribution
df["Amount"].hist(bins=50)
plt.title("Amount Distribution")
plt.show()

# Fraud vs Non-Fraud
sns.boxplot(x="Class", y="Amount", data=df)
plt.show()

# Correlation
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), cmap="coolwarm")
plt.show()