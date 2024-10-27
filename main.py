from sklearn.datasets import fetch_california_housing
import pandas as pd

data = fetch_california_housing()
dataframe = pd.DataFrame(data.data, columns=data.feature_names)
dataframe['price'] = data.target