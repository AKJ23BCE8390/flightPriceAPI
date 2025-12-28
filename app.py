from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os
import gdown

app = Flask(__name__)

MODEL_PATH = "flight_price_pipeline.pkl"
GDRIVE_FILE_ID = "1YQ1o_5aBotORxSHFGuRcSNHFLiGMPPH_"


# Download model if not exists
if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive...")
    url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
    gdown.download(url, MODEL_PATH, quiet=False)

# Load trained pipeline
model = joblib.load(MODEL_PATH)


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Flight Price Prediction API is running 🚀"
    })


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()

        # Convert JSON to DataFrame
        df = pd.DataFrame([data])

        # Make prediction
        prediction = model.predict(df)[0]

        return jsonify({
            "predicted_price": round(float(prediction), 2)
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
