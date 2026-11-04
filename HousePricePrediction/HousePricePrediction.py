#Importing Libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

#  Load the Dataset
data = pd.read_csv("HousePricePrediction/house_price_prediction_dataset (2).csv")
df = pd.DataFrame(data)

# Check whether the null data is present or not.
print(df.isnull().sum())

# Separating Features and Target
X = df.drop("house_price",axis=1)
y = df["house_price"]

# Using the Standard Scaler 
Scaler = StandardScaler()
X = Scaler.fit_transform(X)
# spliting of the datasets
X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.8)
# Declaration of the model
model = XGBRegressor(n_estimators=500, max_depth= 5, learning_rate = 0.03,min_child_weight= 3,random_state= 42)
model.fit(X_train,y_train)
# Prediction of the model 
prediction = model.predict(X_test)
print(prediction)


import numpy as np

# Predictions
y_pred = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))

r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("RMSE:", rmse)
print("R2 Score:", r2)
