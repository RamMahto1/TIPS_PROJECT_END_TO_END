import pandas as pd
from src.pipeline.predict_pipeline import PredictPipeline

sample_input = pd.DataFrame({
    "total_bill": [18.43],
    "tip": [3.00],
    "sex": ["Male"],
    "smoker": ["No"],
    "day": ["Sun"],
    "time": ["Dinner"],
    "size": [4]
})

predictor = PredictPipeline()
result = predictor.predict(sample_input)

print("Prediction Result:", result)
