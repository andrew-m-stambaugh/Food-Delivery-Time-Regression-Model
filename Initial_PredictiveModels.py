# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

# Read in the clean training data
train = pd.read_csv('train_clean.csv')

# Visualize the distributions of time taken with regard to different features
features = ['Sunny', 'Fog', 'month_even', 'Urban_Density', 'Road_traffic_density', 
            'Vehicle_condition', 'multiple_deliveries', 'Festival', 
            'Delivery_person_Age', 'Delivery_person_Ratings']
for feature in features:
    sns.boxplot(x=feature, y='Time_taken(min)', data=train)
    plt.show()

# Remove suburban deliveries
train = train[train['Urban_Density'] != 1].reset_index(drop=True)

# Feature Engineering
train['Delivery_person_Age'] = (train['Delivery_person_Age'] >= 30).astype(int)
train['Geo_Distance < 10'] = (train['Geo_Distance'] < 10).astype(int)
train['Geo_Distance >= 10'] = (train['Geo_Distance'] >= 10).astype(int)
train['Delivery_person_Ratings'] = (train['Delivery_person_Ratings'] >= 4.5).astype(int)
train['Vehicle_condition'] = (train['Vehicle_condition'] > 0).astype(int)

# Drop the "Geo_Distance" feature from the dataset
train = train.drop("Geo_Distance", axis=1)

# Split the dataset into training and testing sets
x = train.drop("Time_taken(min)", axis=1).iloc[:, 1:]
y = train["Time_taken(min)"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/3, random_state=1)


#%% Train Random Forest Regression model and plot results

# Train RFR with parameters
rf = RandomForestRegressor(
    max_depth=11,
    n_estimators=500,
    random_state=317,
    max_features=8
).fit(x_train, y_train)

# Use the trained model to predict on test data
y_pred = rf.predict(x_test)

# Calculate accuracy of the model using r2_score and display results
accuracy = r2_score(y_test, y_pred) * 100
print(f"Accuracy of the Random Forest model is {accuracy:.2f}")

# Plot predictions against true values
plt.scatter(y_test, y_pred, marker='.')
plt.plot([7, 20, 30, 40, 50, 55], [7, 20, 30, 40, 50, 55], color='red')
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('Predictions vs True Values, Random Forest')
plt.show()

# Plot the residuals
plt.scatter(y_test, y_pred - y_test, marker='.')
plt.xlabel('y')
plt.ylabel('Residuals')
plt.title('Residuals')
plt.show()

#%% Train and plot FFNN

# Convert data to numpy arrays and normalize with MinMaxScaler


x_train_np = MinMaxScaler().fit_transform(x_train.to_numpy().astype('float32'))
x_test_np = MinMaxScaler().fit_transform(x_test.to_numpy().astype('float32'))
y_train_np = MinMaxScaler().fit_transform(y_train.to_numpy().astype('float32').reshape(-1, 1))

# Define the neural network model
def create_nn_model():
    model = Sequential()
    model.add(Dense(units=25, input_dim=16, activation='relu'))
    model.add(Dense(units=25, input_dim=15, activation='relu'))
    model.add(Dense(units=25, activation='relu', kernel_initializer='normal'))
    model.add(Dense(units=1, kernel_initializer='normal'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

# Train the neural network model using KerasRegressor
fnn = KerasRegressor(build_fn=create_nn_model, epochs=30, batch_size=5)
fnn.fit(x_train_np, y_train_np)

# Use the trained model to predict on test data
y_pred = fnn.predict(x_test_np)

# Calculate accuracy of the model using r2_score and display results
accuracy = r2_score(y_test.reset_index(drop=True), y_pred) * 100
print(f"Accuracy of the model is {accuracy:.2f}")

# Plot predictions against true values
plt.scatter(y_test, y_pred)
plt.plot([7, 20, 30, 40, 50, 55], [7, 20, 30, 40, 50, 55], color='red')
plt.xlabel('True Value of Y')
plt.ylabel('Predictions')
plt.title('Predictions vs True Values, FNN')
plt.show()
