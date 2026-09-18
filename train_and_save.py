import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering

def train_and_save_model():
    data_path = os.path.join(os.path.dirname(__file__), "smartcart_customers.csv")
    df = pd.read_csv(data_path)

    # 1. Missing values
    df["Income"] = df["Income"].fillna(df["Income"].median())

    # 2. Feature engineering
    df["Age"] = 2026 - df["Year_Birth"]
    df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], dayfirst=True)
    reference_date = df["Dt_Customer"].max()
    df["Customer_Tenure_Days"] = (reference_date - df["Dt_Customer"]).dt.days
    df["Total_spending"] = df["MntWines"] + df["MntFruits"] + df["MntMeatProducts"] + df["MntFishProducts"] + df["MntSweetProducts"] + df["MntGoldProds"]
    df["Total_children"] = df["Kidhome"] + df["Teenhome"]

    df["Education"] = df["Education"].replace({
        "Basic": "Undergraduate", 
        "2n Cycle": "Undergraduate",
        "Master": "Postgraduate",
        "PhD": "Postgraduate",
        "Graduation": "Graduate"
    })

    df["Living_With"] = df["Marital_Status"].replace({
        "Married": "Partner", 
        "Together": "Partner",
        "Single": "Alone",
        "Divorced": "Alone",
        "Widow": "Alone",
        "Alone": "Alone",
        "Absurd": "Alone",
        "YOLO": "Alone"
    })

    cols_to_drop = ["ID", "Year_Birth", "Marital_Status", "Kidhome", "Teenhome", "Dt_Customer", "MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts", "MntSweetProducts", "MntGoldProds"]
    data_cleaned = df.drop(columns=cols_to_drop)

    # Outliers
    data_cleaned = data_cleaned[(data_cleaned["Age"] < 90)]
    data_cleaned = data_cleaned[(data_cleaned["Income"] < 600000)]

    # Encoding
    cat_cols = ["Education", "Living_With"]
    ohe = OneHotEncoder(sparse_output=False)
    enc_cols = ohe.fit_transform(data_cleaned[cat_cols])
    enc_df = pd.DataFrame(enc_cols, columns=ohe.get_feature_names_out(cat_cols), index=data_cleaned.index)
    data_encoded = pd.concat([data_cleaned.drop(columns=cat_cols), enc_df], axis=1)

    feature_names = data_encoded.columns.tolist()

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data_encoded)

    # PCA
    pca = PCA(n_components=3, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    # Clustering
    agg = AgglomerativeClustering(n_clusters=4, linkage="ward")
    cluster_labels = agg.fit_predict(X_pca)

    # Compute PCA Centroids for each cluster
    centroids = {}
    for c in range(4):
        centroids[c] = np.mean(X_pca[cluster_labels == c], axis=0)

    # Cluster characteristics summaries
    data_encoded["cluster"] = cluster_labels
    cluster_profiles = data_encoded.groupby("cluster").mean().to_dict(orient="index")

    model_artifacts = {
        "scaler": scaler,
        "pca": pca,
        "ohe": ohe,
        "feature_names": feature_names,
        "centroids": centroids,
        "cluster_profiles": cluster_profiles
    }

    model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model_artifacts, f)

    print("Model trained and saved to model.pkl successfully.")
    return model_artifacts

if __name__ == "__main__":
    train_and_save_model()
