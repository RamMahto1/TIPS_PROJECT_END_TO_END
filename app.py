from flask import Flask, request, render_template
import pickle
import pandas as pd

app = Flask(__name__)

def load_obj(file_path):
    with open(file_path, "rb") as file_obj:
        return pickle.load(file_obj)

# Dummy model and preprocessor
model = load_obj("artifacts/model.pkl")
preprocessor = load_obj("artifacts/preprocessor.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = {
            'total_bill': [float(request.form['total_bill'])],
            'sex': [request.form['sex']],
            'smoker': [request.form['smoker']],
            'day': [request.form['day']],
            'time': [request.form['time']],
            'size': [int(request.form['size'])]
        }

        input_df = pd.DataFrame(data)
        input_scaled = preprocessor.transform(input_df)
        prediction = model.predict(input_scaled)

        result = f"Prediction: {'Tip Given' if prediction[0] == 1 else 'No Tip'}"
        return render_template('index.html', prediction_text=result)

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
