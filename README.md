# Inventory Demand Forecasting
This project implements a machine learning–based demand forecasting system that predicts daily product demand using historical retail sales data.

# Web App Access
The deployed application can be found at these URLs: 
- AWS: [retail-forecast-env.eba-apyigjrc.ap-south-1.elasticbeanstalk.com](http://retail-forecast-env.eba-apyigjrc.ap-south-1.elasticbeanstalk.com/)
- Railway: [retail-forecast-api-production.up.railway.app](https://retail-forecast-api-production.up.railway.app/)

(Note: The AWS app uses HTTP and not HTTPS, so the website might not be accessible on some mobile browsers)

# Methodology:
- Trains a Random Forest Regressor on daily sales data.
- Uses lag, rolling, and calendar features to capture demand patterns.
- Provides a Flask API endpoint for single-day demand prediction.
- Is deployed on AWS Elastic Beanstalk for cloud-based access.


# Tech stack
| Layer           | Tools                 |
| --------------- | --------------------- |
| Language        | Python                |
| Data Processing | Pandas, NumPy         |
| Modeling        | Scikit-learn          |
| Visualization   | Matplotlib            |
| Backend         | Flask                 |
| Deployment      | AWS Elastic Beanstalk |


# Note
The model was trained only on sales data from 1 Jan 2023 to 1 Jan 2024. Predictions beyond this range are unsupported.
It relies on historical patterns like previous days’ sales, 7-day and 28-day averages to make predictions.
When dates go beyond that range, there’s no recent data to compute these features, so the model cannot generate reliable forecasts.
