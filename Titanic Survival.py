import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Loads Dataset
the_url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
titanic = pd.read_csv(the_url)
print("Dataset Shape:", titanic.shape)
print("\nDataset Info:")
print(titanic.info())
print("\nFirst 5 Rows:")
print(titanic.head())


# Missing Values
print("\nMissing Values:")
print(titanic.isnull().sum())
plt.figure(figsize=(8,6))
sns.heatmap(titanic.isnull(), cbar=False, cmap="viridis")
plt.title("Missing Values Heatmap")
plt.show()


# Survival Counts
plt.figure(figsize=(6,4))
sns.countplot(x="Survived", data=titanic, hue="Survived", palette="Set2", legend=False)
plt.title("Survival Counts (0 = No, 1 = Yes)")
plt.show()


# 4. Survival by Gender
plt.figure(figsize=(6,4))
sns.countplot(x="Sex", hue="Survived", data=titanic, palette="pastel")
plt.title("Survival by Gender")
plt.show()


# 5. Survival by Passenger Class
plt.figure(figsize=(6,4))
sns.countplot(x="Pclass", hue="Survived", data=titanic, palette="muted")
plt.title("Survival by Passenger Class")
plt.show()


# 6. Age Distribution
plt.figure(figsize=(8,6))
sns.histplot(titanic["Age"].dropna(), kde=True, bins=30, color="blue")
plt.title("Age Distribution of Passengers")
plt.show()


# 7. Correlation Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(titanic.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()
