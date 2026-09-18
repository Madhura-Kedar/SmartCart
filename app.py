from flask import Flask, render_template, request
import os
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")

# Train model if not already saved
if not os.path.exists(MODEL_PATH):
    from train_and_save import train_and_save_model
    train_and_save_model()

with open(MODEL_PATH, "rb") as f:
    artifacts = pickle.load(f)

scaler = artifacts["scaler"]
pca = artifacts["pca"]
centroids = artifacts["centroids"]
feature_names = artifacts["feature_names"]

# Actual Cluster Metadata based on SmartCart.ipynb Analysis
CLUSTER_METADATA = {
    0: {
        "name": "Budget Partnered Families",
        "description": "Lower income households living with a partner. Characterized by higher number of children, conservative spending habits, and moderate store & web activity.",
        "avg_income": "$39,681",
        "avg_spending": "$222",
        "avg_children": "1.24",
        "living_status": "100% Partnered",
        "badge_class": "badge-c0",
        "accent_color": "#f43f5e"
    },
    1: {
        "name": "Affluent Partnered High-Spenders",
        "description": "High-earning partnered customers with few children. Characterized by strong purchasing power across store and catalog channels with large total cart values.",
        "avg_income": "$72,808",
        "avg_spending": "$1,237",
        "avg_children": "0.51",
        "living_status": "100% Partnered",
        "badge_class": "badge-c1",
        "accent_color": "#818cf8"
    },
    2: {
        "name": "Budget Single Individuals",
        "description": "Lower income single/divorced/widowed individuals with children at home. Low overall spending, frequent web visits, and price-sensitive purchase patterns.",
        "avg_income": "$36,960",
        "avg_spending": "$166",
        "avg_children": "1.27",
        "living_status": "99.3% Alone",
        "badge_class": "badge-c2",
        "accent_color": "#fbbf24"
    },
    3: {
        "name": "Affluent Single High-Responders",
        "description": "High income single customers with high spending and few children. Highest responsiveness to marketing campaigns (32.0%) and frequent web/catalog buyers.",
        "avg_income": "$70,723",
        "avg_spending": "$1,190",
        "avg_children": "0.46",
        "living_status": "100% Alone",
        "badge_class": "badge-c3",
        "accent_color": "#34d399"
    }
}

@app.route('/')
def home():
    return render_template('index.html', page_title="Home")

@app.route('/analysis')
def analysis():
    return render_template('analysis.html', page_title="ML Analysis & EDA")

@app.route('/segmentation', methods=['GET', 'POST'])
def segmentation():
    prediction_result = None
    user_inputs = {
        "income": 58000,
        "age": 45,
        "education": "Graduate",
        "living_with": "Partner",
        "total_spending": 850,
        "total_children": 1,
        "recency": 48,
        "tenure_days": 350,
        "web_purchases": 5,
        "catalog_purchases": 3,
        "store_purchases": 6,
        "deals_purchases": 2,
        "web_visits": 5,
        "complain": 0,
        "response": 0
    }

    if request.method == 'POST':
        try:
            user_inputs = {
                "income": float(request.form.get("income", 58000)),
                "age": float(request.form.get("age", 45)),
                "education": request.form.get("education", "Graduate"),
                "living_with": request.form.get("living_with", "Partner"),
                "total_spending": float(request.form.get("total_spending", 850)),
                "total_children": float(request.form.get("total_children", 1)),
                "recency": float(request.form.get("recency", 48)),
                "tenure_days": float(request.form.get("tenure_days", 350)),
                "web_purchases": float(request.form.get("web_purchases", 5)),
                "catalog_purchases": float(request.form.get("catalog_purchases", 3)),
                "store_purchases": float(request.form.get("store_purchases", 6)),
                "deals_purchases": float(request.form.get("deals_purchases", 2)),
                "web_visits": float(request.form.get("web_visits", 5)),
                "complain": float(request.form.get("complain", 0)),
                "response": float(request.form.get("response", 0))
            }

            # One-Hot Encoding
            edu = user_inputs["education"]
            living = user_inputs["living_with"]
            edu_grad = 1.0 if edu == "Graduate" else 0.0
            edu_post = 1.0 if edu == "Postgraduate" else 0.0
            edu_under = 1.0 if edu == "Undergraduate" else 0.0
            living_alone = 1.0 if living == "Alone" else 0.0
            living_partner = 1.0 if living == "Partner" else 0.0

            # Construct 18-feature row matching exact model feature ordering
            raw_vector = [
                user_inputs["income"],
                user_inputs["recency"],
                user_inputs["deals_purchases"],
                user_inputs["web_purchases"],
                user_inputs["catalog_purchases"],
                user_inputs["store_purchases"],
                user_inputs["web_visits"],
                user_inputs["complain"],
                user_inputs["response"],
                user_inputs["age"],
                user_inputs["tenure_days"],
                user_inputs["total_spending"],
                user_inputs["total_children"],
                edu_grad,
                edu_post,
                edu_under,
                living_alone,
                living_partner
            ]

            df_vector = pd.DataFrame([raw_vector], columns=feature_names)

            # Pipeline execution: Scaling -> 3D PCA -> Nearest Centroid Distance
            scaled_vector = scaler.transform(df_vector)
            pca_vector = pca.transform(scaled_vector)[0]

            distances = {c: float(np.linalg.norm(pca_vector - centroids[c])) for c in range(4)}
            predicted_cluster_id = min(distances, key=distances.get)

            prediction_result = {
                "cluster_id": predicted_cluster_id,
                "metadata": CLUSTER_METADATA[predicted_cluster_id],
                "pca_coords": [round(float(c), 3) for c in pca_vector],
                "distances": {k: round(v, 3) for k, v in distances.items()}
            }
        except Exception as e:
            prediction_result = {"error": str(e)}

    return render_template(
        'segmentation.html', 
        page_title="Customer Segmentation", 
        inputs=user_inputs, 
        result=prediction_result
    )

if __name__ == '__main__':
    app.run(debug=True, port=5000)
