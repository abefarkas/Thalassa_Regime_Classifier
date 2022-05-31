FROM python:3.8.6-buster

WORKDIR /app

COPY Thalassa_Regime_Classifier .
COPY arima_fitted.joblib .
COPY requirements.txt .

RUN pip install -r requirements.txt
