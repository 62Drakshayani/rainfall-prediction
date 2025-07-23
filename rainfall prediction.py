# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Step 2: Import Data Set
file_path = 'Data set - Rainfall Prediction.csv'  # Make sure the file path is correct
rainfall_data = pd.read_csv(file_path)

# Step 3: Handle Null Values
# Replacing null values in monthly rainfall data with the mean values for those months
for month in ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']:
    rainfall_data[month].fillna(rainfall_data[month].mean(), inplace=True)

# Recalculating the combined seasonal data and annual rainfall
rainfall_data['Jan-Feb'] = rainfall_data['JAN'] + rainfall_data['FEB']
rainfall_data['Mar-May'] = rainfall_data['MAR'] + rainfall_data['APR'] + rainfall_data['MAY']
rainfall_data['Jun-Sep'] = rainfall_data['JUN'] + rainfall_data['JUL'] + rainfall_data['AUG'] + rainfall_data['SEP']
rainfall_data['Oct-Dec'] = rainfall_data['OCT'] + rainfall_data['NOV'] + rainfall_data['DEC']
rainfall_data['ANNUAL'] = rainfall_data.loc[:, 'JAN':'DEC'].sum(axis=1)

# Step 4: Categorical to Numerical
label_encoder = LabelEncoder()
rainfall_data['SUBDIVISION'] = label_encoder.fit_transform(rainfall_data['SUBDIVISION'])

# Step 5: Features & Target
X = rainfall_data.drop('ANNUAL', axis=1)
y = rainfall_data['ANNUAL']

# Step 6: Train & Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Building
random_forest_model = RandomForestRegressor(n_estimators=100, random_state=42)
random_forest_model.fit(X_train, y_train)

# Prediction
y_pred = random_forest_model.predict(X_test)

# Step 7: Performance Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Displaying the performance metrics
print("Mean Absolute Error:", mae)
print("Root Mean Squared Error:", rmse)
print("R-squared Value:", r2)
