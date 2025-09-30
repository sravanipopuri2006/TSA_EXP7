# Ex.No: 07                                       AUTO REGRESSIVE MODEL
### Date: 30.09.2025



### AIM:
To Implementat an Auto Regressive Model using Python
### ALGORITHM:
1. Import necessary libraries
2. Read the CSV file into a DataFrame
3. Perform Augmented Dickey-Fuller test
4. Split the data into training and testing sets.Fit an AutoRegressive (AR) model with 13 lags
5. Plot Partial Autocorrelation Function (PACF) and Autocorrelation Function (ACF)
6. Make predictions using the AR model.Compare the predictions with the test data
7. Calculate Mean Squared Error (MSE).Plot the test data and predictions.
### PROGRAM
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv('/content/Gold Price (2013-2023).csv', parse_dates=['Date'], index_col='Date')


for col in ['Price', 'Open', 'High', 'Low']:
    data[col] = data[col].astype(str).str.replace(',', '').astype(float)

gold_prices = data[['Price']]


result = adfuller(gold_prices['Price'].dropna())
print('ADF Statistic:', result[0])
print('p-value:', result[1])


train_size = int(len(gold_prices) * 0.8)
train_data, test_data = gold_prices[0:train_size], gold_prices[train_size:]

model = AutoReg(train_data['Price'], lags=13)
ar_model_fit = model.fit()

plt.figure(figsize=(10, 5))
plot_acf(train_data['Price'], lags=40)
plt.title('Autocorrelation Function (ACF) - Gold Price')
plt.show()

plt.figure(figsize=(10, 5))
plot_pacf(train_data['Price'], lags=40)
plt.title('Partial Autocorrelation Function (PACF) - Gold Price')
plt.show()

predictions = ar_model_fit.predict(start=len(train_data), end=len(train_data) + len(test_data) - 1, dynamic=False)


plt.figure(figsize=(12, 6))
plt.plot(test_data.index, test_data, label='Actual Test Data', color='blue')
plt.plot(test_data.index, predictions, label='Predictions', color='red', linestyle='dashed')
plt.title('Actual vs Predicted Gold Prices (AR Model)')
plt.xlabel('Date')
plt.ylabel('Gold Price')
plt.legend()
plt.grid(True)
plt.show()


mse = mean_squared_error(test_data, predictions)
print('Mean Squared Error (MSE):', mse)

```
### OUTPUT:

<img width="593" height="492" alt="image" src="https://github.com/user-attachments/assets/3c43cd41-138e-44f4-bad4-d78b30ae7ba4" />
<img width="571" height="448" alt="image" src="https://github.com/user-attachments/assets/977abec3-0411-4247-91ff-c613a15d171f" />
<img width="1037" height="563" alt="image" src="https://github.com/user-attachments/assets/7a071bc7-7011-4b86-9a16-36864ca456a4" />




### RESULT:
Thus we have successfully implemented the auto regression function using python.
