#  Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

#  Load the Dataset
df = pd.read_csv('Advertising.csv')

#  Clean Data
# Drop 'Unnamed: 0' column if it exists (index column)
if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])

#  Check for missing values
print("Missing values:\n", df.isnull().sum())

#  Correlation Heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

#  Feature-wise Scatter Plots
fig, axs = plt.subplots(1, 3, figsize=(15, 4))
sns.scatterplot(x='TV', y='Sales', data=df, ax=axs[0])
sns.scatterplot(x='Radio', y='Sales', data=df, ax=axs[1])
sns.scatterplot(x='Newspaper', y='Sales', data=df, ax=axs[2])
axs[0].set_title("TV vs Sales")
axs[1].set_title("Radio vs Sales")
axs[2].set_title("Newspaper vs Sales")
plt.tight_layout()
plt.show()

#  Features and Target
X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']

# 🔀 Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#  Train the Model
model = LinearRegression()
model.fit(X_train, y_train)

#  Make Predictions
y_pred = model.predict(X_test)

#  Evaluate the Model
print("\nModel Performance:")
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

#  Feature Importance
coeff_df = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
print("\nFeature Importance:\n", coeff_df)

#  Actual vs Predicted Sales Plot (with y = x line)
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, color='blue', edgecolor='k', alpha=0.7)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.grid(True)

# Ideal prediction line
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Ideal Prediction')

plt.legend()
plt.show()
