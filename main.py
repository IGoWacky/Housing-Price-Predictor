from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import pandas as pd

data = fetch_california_housing()
dataframe = pd.DataFrame(data.data, columns=data.feature_names)
dataframe['price'] = data.target

dataframe = dataframe.drop(columns=['Population'])

x = dataframe.drop(columns=['price']) #features to be analyzed
y = dataframe['price'] #target variable
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=50)

model = RandomForestRegressor(n_estimators=100, max_depth=100, random_state=50)
model.fit(x_train, y_train) #this trains the model and the previous line initializes it
y_prediction = model.predict(x_test)

mAbsErr = mean_absolute_error(y_test, y_prediction)
rmSqErr = root_mean_squared_error(y_test, y_prediction)