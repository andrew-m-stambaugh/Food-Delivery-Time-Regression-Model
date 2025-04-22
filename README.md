# 🛵 Food Delivery Time Regression Model

This repository contains files used and created as part of a predictive modeling challenge on Kaggle:  
**[Food Delivery Dataset on Kaggle](https://www.kaggle.com/datasets/gauravmalik26/food-delivery-dataset?select=train.csv)**

The goal was to build a model that accurately predicts delivery times based on order details, geography, time of day, and delivery conditions.

---

## 📘 Background

This project was initially completed as part of **MECH_ENG 329: Mechanistic Data Science for Engineering** at Northwestern University in 2022. It was one of my first end-to-end machine learning projects, and while many decisions now feel rudimentary, it was a foundational learning experience that built my interest in applied modeling.

I recently revisited the project with my current experience as a data analyst to improve the model, with a strong focus on clean, simple, but effective code.
👉 View the updated modeling notebook here: [updated_predictive_models.ipynb](./updated_predictive_models.ipynb)  

---

## 📁 Files

- `Initial Final Report.pdf`  
  My original college report. It documents the initial model development, context, and results using Random Forest Regression and a simple feed-forward neural network (FNN). I have not changed this report since it was submitted as I prefer to maintain its authenticity from the pre-ChatGPT era.
  
  **R² Scores:**  
  - Random Forest: **0.73**  
  - Feed-Forward Neural Net: **0.73**

- `initial_data_cleaning.py`  
  Script used to clean the raw Kaggle dataset. Includes:
  - Null value removal
  - Timestamp parsing
  - Geographic distance calculation
  - Categorical encoding
  - Temporal feature creation (e.g., day of week, month)

- `initial_predictive_models.py`  
  Initial model training script. Includes:
  - Binary encoding of discontinuous numerical features
  - Model training and evaluation for both RF and FNN

- `initial_train_clean.csv`  
  Cleaned dataset used for modeling. Includes 15 features such as:
  - Driver Age, Driver Rating
  - Traffic Level, Vehicle Condition
  - Festival, Weather Flags, Geo Distance
  - Urban density, Time of day, Rush hour flags
  - A fun surprise: **"Even Month?"**, which surprisingly showed high correlation

- `updated_predictive_models.ipynb` / `FollowUpAnalysis.html`  
  **Updated analysis (2025)** — A complete overhaul of the original pipeline to better outline the entire process and improve the model:
  - Enhanced data cleaning and transformation
  - Model comparison with **XGBoost**, **Random Forest**, and **PyTorch Feed-Forward Neural Network**
  - Feature importance visualizations and tuning
  - Significantly improved R² scores over the original models, 0.86 for the XGBoost model.

---

## 🚀 Improvements Over the Original

- Cleaned all time fields using true datetime parsing  
- Added geospatial decomposition: `lat_diff`, `lon_diff`, `lat_bucket`  
- Modeled time-based behaviors with `order_hour`, `rush hour`, `weekend`, etc.  
- Used **GridSearchCV** to tune model hyperparameters (full search not in the notebook file)  
- Overall, less mistakes were made given my greater experience and better current coding habits.

---

## 💬 Reflections

This project is a good snapshot of how my skills have evolved — from understanding the basics of scikit-learn to leveraging structured pipelines and advanced models for better performance. While the original project was valuable, this update reflects a much more production-minded approach to data science.
