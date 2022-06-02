FROM python:3.8.6-buster

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY Thalassa_Regime_Classifier Thalassa_Regime_Classifier
COPY *.joblib .

COPY start.sh .

CMD /app/start.sh
