# Text-to-Emotion
to run model

cd api
uvicorn main:app --reload
http POST http://127.0.0.1:8000/predict filename="zoomcity(by_3lines).txt"
