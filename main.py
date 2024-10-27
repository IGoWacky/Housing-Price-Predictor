from sklearn.datasets import fetch_california_housing as housing
from sklearn.model_selection import train_test_split as testSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from flask import Flask, request, render_template, jsonify
import pandas as pd

data = housing()
dataframe = pd.DataFrame(data.data, columns=data.feature_names)
dataframe['price'] = data.target

dataframe = dataframe.drop(columns=['Population'])

x = dataframe.drop(columns=['price']) #features to be analyzed
y = dataframe['price'] #target variable
x_train, x_test, y_train, y_test = testSplit(x, y, test_size=0.2, random_state=50)

model = RandomForestRegressor(n_estimators=100, max_depth=100, random_state=50)
model.fit(x_train, y_train) #this trains the model and the previous line initializes it

app = Flask(__name__)
@app.route('/')
def form():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    med_inc = float(request.form['med_inc'])
    house_age = float(request.form['house_age'])
    ave_rooms = float(request.form['ave_rooms'])
    ave_bedrooms = float(request.form['ave_bedrooms'])
    ave_occupancy = float(request.form['ave_occupancy'])
    latitude = float(request.form['latitude'])
    longitude = float(request.form['longitude'])

    user_data = pd.DataFrame({
        'MedInc': [med_inc],
        'HouseAge': [house_age],
        'AveRooms': [ave_rooms],
        'AveBedrms': [ave_bedrooms],
        'AveOccup': [ave_occupancy],
        'Latitude': [latitude],
        'Longitude': [longitude]
    })

    prediction = model.predict(user_data)
    predicted_price = round(prediction[0], 2)
    return render_template('index.html', prediction_text=f"Predicted House Price: ${round(predicted_price, 2)}")

if __name__ == "__main__":
    app.run(debug=True)