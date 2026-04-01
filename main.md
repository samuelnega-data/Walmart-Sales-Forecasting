### Surface Level EDA
This section performs an initial exploration of the dataset to understand its structure, distributions, and basic statistical properties before applying any transformations or models.

Lets simply explore the dataset in question, this will give us a strong baseline understanding of the stastical measure of the dataset.

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
```

```python
df = pd.read_csv()
```

```python
df.head(5)
```
<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Store</th>
      <th>Date</th>
      <th>Weekly_Sales</th>
      <th>Holiday_Flag</th>
      <th>Temperature</th>
      <th>Fuel_Price</th>
      <th>CPI</th>
      <th>Unemployment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>05-02-2010</td>
      <td>1643690.90</td>
      <td>0</td>
      <td>42.31</td>
      <td>2.572</td>
      <td>211.096358</td>
      <td>8.106</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>12-02-2010</td>
      <td>1641957.44</td>
      <td>1</td>
      <td>38.51</td>
      <td>2.548</td>
      <td>211.242170</td>
      <td>8.106</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>19-02-2010</td>
      <td>1611968.17</td>
      <td>0</td>
      <td>39.93</td>
      <td>2.514</td>
      <td>211.289143</td>
      <td>8.106</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>26-02-2010</td>
      <td>1409727.59</td>
      <td>0</td>
      <td>46.63</td>
      <td>2.561</td>
      <td>211.319643</td>
      <td>8.106</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>05-03-2010</td>
      <td>1554806.68</td>
      <td>0</td>
      <td>46.50</td>
      <td>2.625</td>
      <td>211.350143</td>
      <td>8.106</td>
    </tr>
  </tbody>
</table>
</div>

```python
df.describe()
```

<!-- table stays same -->

```python
df.shape
```

```python
df.isna().sum()
```

```python
df.dtypes
```

```python
weekly_sales = df.groupby('Date', as_index=False)['Weekly_Sales'].sum()

plt.figure(figsize=(12,6))
plt.plot(weekly_sales['Date'], weekly_sales['Weekly_Sales'], c = 'r')
plt.title('Weekly Sales')
plt.xlabel('Date')
plt.ylabel('Sales Amount')
```

![Weekly Sales](https://github.com/user-attachments/assets/4af1ff90-1e8a-4639-b4ec-3b76b66fb222)

```python
df2 = df.copy()

df2 = df2.drop(columns='Date')

corr_value = df2.corr().round(2)

plt.figure(figsize=(12,5))
plt.title('Correlation HeatMap')
sns.heatmap(df2.corr(), cmap='Blues', annot=corr_value)
```

![Heatmap](https://github.com/user-attachments/assets/454ad5dd-2575-40bf-8a97-f133e5340c06)

```python
hol_flag = df.groupby('Holiday_Flag').agg({
    'Holiday_Flag': 'count',
    'Weekly_Sales': 'sum'
})

total_sales = hol_flag['Weekly_Sales'].sum()
total_sale_lines = len(df)

hol_flag['Percentage_Cont'] = (hol_flag['Weekly_Sales'] / total_sales)*100
hol_flag['Sale_Line_Percent'] = (hol_flag['Holiday_Flag'] / total_sale_lines)*100

hol_flag
```

#### Takeaways
- The dataset contains **6,435 records across 8 features**, with no missing values, indicating clean data.
- Weekly sales vary significantly (high standard deviation), suggesting strong variability across stores and time.
- Holiday weeks represent a **small portion of total records (~7%)**, but contribute a meaningful share of total sales.
- The time series plot suggests **clear trends and possible seasonality** in weekly sales.
- Correlation heatmap shows that **no single external factor (temperature, fuel price, CPI, unemployment)** strongly drives sales alone.

---

#### Feature Enginerring
This section creates additional time-based and lag-based features to capture temporal patterns and improve model performance.

Lets build some additional features ontop of our dataset to enhance our correlations 

```python
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
```

![Correlation Heatmap](https://github.com/user-attachments/assets/4bb16159-be72-4a89-95c3-e392d6e33ea1)

#### Takeaways
- Lag features (Sales_Lag_1, Sales_Lag_2) show **strong correlation with current sales**, confirming temporal dependency.
- Rolling averages help smooth noise and capture **short-term trends**.
- Time-based features (month, quarter, etc.) enable detection of **seasonality patterns**.
- Overall, feature engineering significantly improves the dataset’s predictive potential.

---

#### Building Linear Regression Model
This section applies a linear regression model using engineered features to predict weekly sales and evaluates its performance.

```python
df2 = df[['Sales_Lag_1', 'Sales_Lag_2', 'MA_3_Week_Lagged', 'Store','Weekly_Sales']]
df2.dropna(inplace=True)
df2.shape

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
```

![Predictions vs Actual](https://github.com/user-attachments/assets/5d20e7dd-d802-4cb6-913b-bfa2fa05c360)

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats
import numpy as np

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
```

#### Takeaways
- The model achieves a **high R² (~0.91)**, indicating strong predictive performance.
- Lag features are highly effective, confirming that **past sales are strong predictors of future sales**.
- Errors (MAE ~85k, RMSE ~164k) show some deviation, especially for extreme values.
- Residuals are **not normally distributed (p ≈ 0)**, suggesting the model may miss some patterns.
- The model performs well overall but could be improved using **non-linear models or additional features**.

---
