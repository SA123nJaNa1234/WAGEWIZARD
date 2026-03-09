from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

PIPELINE_PATH = "data/models/salary_pipeline.pkl"

# Load pipeline
try:
    pipeline = joblib.load(PIPELINE_PATH)
    model = pipeline["model"]
    vectorizer = pipeline["vectorizer"]
    scaler = pipeline["scaler"]
    label_encoder = pipeline["label_encoder"]

    print("✓ Pipeline loaded successfully")

except Exception as e:
    print(f"✗ Error loading pipeline: {e}")
    model = None


def build_features(description, experience_level, is_remote, skills_count):
    """
    Build the exact feature vector used during training
    """

    # TF-IDF features (100)
    text_features = vectorizer.transform([description]).toarray()

    # Encode experience level
    exp_encoded = label_encoder.transform([experience_level])[0]

    # Convert boolean
    remote_int = int(is_remote)

    # Numerical features
    numerical = np.array([[skills_count, remote_int, exp_encoded]])

    # Scale numerical features
    numerical_scaled = scaler.transform(numerical)

    # Combine text + numerical
    features = np.hstack([text_features, numerical_scaled])

    return features


@app.route("/")
def home():
    return jsonify({
        "message": "Job Salary Prediction API",
        "endpoints": {
            "/predict": "POST - Single prediction",
            "/predict_batch": "POST - Multiple predictions",
            "/health": "GET - Health check"
        }
    })


@app.route("/predict", methods=["POST"])
def predict():

    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        data = request.get_json()

        description = data.get("description", "")
        experience_level = data.get("experience_level", "Mid")
        is_remote = data.get("is_remote", False)
        skills_count = data.get("skills_count", 3)

        features = build_features(
            description,
            experience_level,
            is_remote,
            skills_count
        )

        prediction = model.predict(features)[0]

        return jsonify({
            "status": "success",
            "predicted_salary": float(prediction),
            "experience_level": experience_level,
            "is_remote": is_remote,
            "skills_count": skills_count
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/predict_batch", methods=["POST"])
def predict_batch():

    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        data = request.get_json()
        jobs = data.get("jobs", [])

        predictions = []

        for job in jobs:

            features = build_features(
                job.get("description", ""),
                job.get("experience_level", "Mid"),
                job.get("is_remote", False),
                job.get("skills_count", 3)
            )

            pred = model.predict(features)[0]

            predictions.append({
                "job_id": job.get("id", "unknown"),
                "predicted_salary": float(pred)
            })

        return jsonify({
            "status": "success",
            "batch_size": len(predictions),
            "predictions": predictions
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/health")
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)