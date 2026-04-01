# ==============================
# Surface Level EDA
# ==============================

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv()

df.head(5)

df.describe()

df.shape

df.isna().sum()

df.dtypes

weekly_sales = df.groupby('Date', as_index=False)['Weekly_Sales'].sum()

plt.figure(figsize=(12,6))
plt.plot(weekly_sales['Date'], weekly_sales['Weekly_Sales'], c='r')
plt.title('Weekly Sales')
plt.xlabel('Date')
plt.ylabel('Sales Amount')
plt.show()

df2 = df.copy()
df2 = df2.drop(columns='Date')

corr_value = df2.corr().round(2)

plt.figure(figsize=(12,5))
plt.title('Correlation HeatMap')
sns.heatmap(df2.corr(), cmap='Blues', annot=corr_value)
plt.show()

hol_flag = df.groupby('Holiday_Flag').agg({
    'Holiday_Flag': 'count',
    'Weekly_Sales': 'sum'
})

total_sales = hol_flag['Weekly_Sales'].sum()
total_sale_lines = len(df)

hol_flag['Percentage_Cont'] = (hol_flag['Weekly_Sales'] / total_sales) * 100
hol_flag['Sale_Line_Percent'] = (hol_flag['Holiday_Flag'] / total_sale_lines) * 100

hol_flag


# ==============================
# Feature Engineering
# ==============================

df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y', errors='coerce')

df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['Week'] = df['Date'].dt.isocalendar().week
df['Day_of_Week'] = df['Date'].dt.dayofweek
df['Month_Name'] = df['Date'].dt.month
df['Quarter'] = df['Date'].dt.quarter
df['Is_Weekend'] = (df['Date'].dt.dayofweek >= 5).astype(int)

df['Sales_Lag_1'] = df['Weekly_Sales'].shift(1)
df['Sales_Lag_2'] = df['Weekly_Sales'].shift(2)
df['MA_3_Week_Lagged'] = df['Weekly_Sales'].shift(1).rolling(3).mean()
df['Rolling_Std_3'] = df['Weekly_Sales'].rolling(3).std()

plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), cmap='Blues', annot=True, fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()


# ==============================
# Machine Learning Model
# ==============================

df2 = df[['Sales_Lag_1', 'Sales_Lag_2', 'MA_3_Week_Lagged', 'Store', 'Weekly_Sales']]
df2.dropna(inplace=True)

X = df2.drop(columns=['Weekly_Sales'], inplace=False)
y = df2['Weekly_Sales']

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

errors = np.abs(y_pred - y_test)

plt.figure(figsize=(12,6))
plt.title('Predictions vs Actual with Error Coloring')
plt.xlabel('Predictions')
plt.ylabel('Actual')

scatter = plt.scatter(y_pred, y_test, c=errors, cmap='coolwarm', alpha=0.7)

cbar = plt.colorbar(scatter)
cbar.set_label('Absolute Error')

plt.show()

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

residuals = y_test - y_pred
residuals_mean = np.mean(residuals)
residuals_std = np.std(residuals)
_, p_value = stats.normaltest(residuals)

print(f"R² Score: {r2:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Residuals Mean: {residuals_mean:.4f}")
print(f"Residuals Std Dev: {residuals_std:.4f}")
print(f"Residuals Normality p-value: {p_value:.4f} (p>0.05 suggests normal residuals)")
