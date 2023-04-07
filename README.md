# Predicting Food Delivery Times with Machine vs Deep Learning

Note: See the final report for a more detailed explanation of the project's process, development, and results. The code in the appendices of the report is outdated, however. Since this project, I have gone back and made the code much more efficient and neat to more properly demonstrate my ability to write neat ML code.

This repository contains all of the files used and created as part of a data science challenge in Kaggle. 
link at https://www.kaggle.com/datasets/gauravmalik26/food-delivery-dataset?select=train.csv. 

FinalReport.pdf: This is the final report that I wrote for the project. It provides all details of the project's context, development, and results. See this pdf if you'd like to look into the project in more detail. The models used were Random Forest Regression and an FNN, and their respective testing accuracies came out to be 73.44% and 72.78%.

DataCleaning.py: This python file contains the code written to properly clean and reformat the raw dataset from Kaggle. This includes dropping the null values, reformatting dates and times to datetime objects, converting geographical coordinates to geodistances, and converting categorical variables to numerical variables. This file also includes the time and date feature engineering (day of the week, month, etc.)

PredictiveModels.py: This python file contains the code written for feature engineering and training the models. The feature engineering includes converting numerical variables with discontinuous and dichotomous behavior to binary variables, and splitting the geodistance data set. The models trained were a Random Forest Regression model and a Feed Forward Neural Network.

train_clean.csv: The data set after being cleaned. There are 15 features, including Driver Age, Driver Rating, Traffic Level, Vehicle Condition, Number of Extra Deliveries, Festival?, Geo Distance, Sunny?, Fog?, Motorcycle?, Urban Density (1-4), Hour, Night?, Rush Hour?, Even Month? (This feature oddly has a strong affect on the data). See the link for the original raw dataset.
