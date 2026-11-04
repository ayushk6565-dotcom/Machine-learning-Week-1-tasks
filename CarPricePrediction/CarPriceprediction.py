import pandas as pd 
import numpy as np
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import(mean_absolute_error,mean_squared_error,r2_score)

data = pd.read_csv("CarPricePrediction/car_price_prediction_dataset.csv")
df = pd.DataFrame(data)
print(df.isnull().sum())
X = df.drop("selling_price_inr",axis=1)
y = df["selling_price_inr"]
Scalar = StandardScaler()
Encoder = OneHotEncoder(handle_unknown="ignore")


X[["km_driven"]] = Scalar.fit_transform(X[["km_driven"]])
X[["brand"]] = Encoder.fit_transform(X[["brand"]])
X[["model"]] = Encoder.fit_transform(X[["model"]])
X[["manufacturing_year"]] = Encoder.fit_transform(X[["manufacturing_year"]])
X[["vehicle_age_years"]] = Encoder.fit_transform(X[["vehicle_age_years"]])
X[["transmission"]] = Encoder.fit_transform(X[["transmission"]])
X[["fuel_type"]] = Encoder.fit_transform(X[["fuel_type"]])
X[["owner_type"]] = Encoder.fit_transform(X[["owner_type"]])

X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.8,random_state=42)
model = RandomForestRegressor(n_estimators=500,max_depth=15,random_state=42,min_samples_split=5)
model.fit(X_train,y_train)
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("MAE:", mae)
print("RMSE:", rmse)
print("R2 Score:", r2)