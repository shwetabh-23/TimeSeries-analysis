## Time-Series Forecasting for Temperature Prediction
# Overview
This project aims to develop time-series forecasting models for temperature prediction. Temperature forecasting is crucial for a wide range of applications, including climate monitoring, energy management, and agriculture. The project explores both univariate and bivariate approaches to leverage historical temperature data collected at 10-minute intervals over a 10-year period.

# Table of Contents
Introduction
Data
Methodology
Models
Evaluation
Usage
Contributing
License 

# Introduction
Temperature prediction plays a significant role in various domains, such as weather forecasting, agriculture, and energy management. Accurate predictions help in making informed decisions and planning for the future. This project focuses on developing time-series forecasting models, specifically utilizing Long Short-Term Memory (LSTM) networks, to predict temperature values.

# Data
The dataset used for this project consists of temperature measurements collected every 10 minutes over a span of 10 years. The dataset is multivariate, containing not only temperature values but also additional relevant features that can enhance the prediction accuracy.

# Data Source: 
Kaggle

# Methodology
Data Preprocessing
Data Cleaning: Handle missing values, outliers, and data anomalies.
Feature Engineering: Create relevant features and transformations for model input.
Model Building
Univariate Approach: Develop models that solely use temperature data for prediction.
Bivariate Approach: Leverage additional features, such as humidity or pressure, in conjunction with temperature data for improved forecasting.
Model Training
Utilize Long Short-Term Memory (LSTM) networks for time-series modeling.
Experiment with other machine learning and deep learning models to compare performance.
Model Evaluation
Assess model performance using appropriate evaluation metrics, e.g., Mean Absolute Error (MAE), Root Mean Squared Error (RMSE).
Utilize time-series cross-validation techniques to evaluate model generalization.
Models
The project includes the implementation of various models:

Univariate LSTM Model: A time-series forecasting model using only temperature data.
Bivariate LSTM Model: Incorporating additional features in addition to temperature data.
Baseline Models: Comparing the LSTM models against traditional time-series forecasting methods.
Evaluation
The success of each model is determined based on its ability to accurately forecast temperature values. Key evaluation metrics include:

Mean Absolute Error (MAE): Measures the average absolute difference between predicted and actual values.
Root Mean Squared Error (RMSE): Provides a measure of the prediction error.


License
Genral Public License 3.0

By creating a robust time-series forecasting model for temperature prediction, this project aims to facilitate informed decision-making in various fields where temperature plays a crucial role. The univariate and bivariate approaches, coupled with LSTM networks and traditional models, offer a comprehensive analysis of forecasting accuracy and performance.




