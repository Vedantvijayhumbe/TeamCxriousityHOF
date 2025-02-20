from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load the trained CNN model
MODEL_PATH = 'tb_cnn_final_model.h5'
model = load_model(MODEL_PATH)

def predict_tb(img_path):
    """Preprocess the image and make a prediction."""
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)[0][0]
    confidence = round(float(prediction) * 100, 2)
    status = "Tuberculosis Detected" if prediction > 0.5 else "No Tuberculosis Detected"

    # Example Key Findings (static or generated based on Grad-CAM in future)
    key_findings = []
    if status == "Tuberculosis Detected":
        key_findings = [
            "Upper lobe infiltrates present",
            "Cavitation observed in right lung",
            "Mediastinal lymphadenopathy detected"
        ]

    # Recommendations based on detection
    recommendations = []
    if status == "Tuberculosis Detected":
        recommendations = [
            "Immediate consultation with pulmonologist",
            "Follow-up PCR test recommended",
            "Start first-line anti-TB therapy"
        ]
    else:
        recommendations = [
            "No immediate action required",
            "Maintain regular health check-ups"
        ]

    return {
        "status": status,
        "confidence": f"{confidence}% Probability",
        "key_findings": key_findings,
        "recommendations": recommendations
    }

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Save and process the uploaded file
    file_path = os.path.join("uploads", file.filename)
    os.makedirs("uploads", exist_ok=True)
    file.save(file_path)

    # Run prediction
    result = predict_tb(file_path)

    # Remove the uploaded file after prediction
    os.remove(file_path)

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
