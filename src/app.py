from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import joblib

# Load the trained model
model = tf.keras.models.load_model("models/best_model.keras")

# Initialize Flask app
app = Flask(__name__)

@app.route("/")
def home():
    return "Heart Disease Prediction API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON data
        data = request.get_json()
        
        # Ensure data is in the correct format
        features = np.array(data["features"]).reshape(1, -1)  # Convert to NumPy array
        
        # Make prediction
        prediction = model.predict(features)
        prediction_label = int(prediction[0][0] > 0.5)  # Convert probability to binary label
        
        return jsonify({"prediction": prediction_label, "probability": float(prediction[0][0])})
    
    except Exception as e:
        return jsonify({"error": str(e)})

# Run the app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
