"""
SmartCart – Customer Segmentation Flask App
==========================================
Routes:
  GET  /          -> renders index.html
  POST /predict   -> returns JSON with cluster_id, segment, and suggestion

Pipeline order (must match training):
  raw inputs -> OHE (Education, Living_With) -> column-aligned DataFrame
             -> scaler.transform -> pca.transform -> kmeans.predict
"""

import json
import joblib
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = Flask(__name__)

BASE_DIR = Path(__file__).parent

# ---------------------------------------------------------------------------
# Cluster metadata  <- edit these to match your actual cluster analysis
# ---------------------------------------------------------------------------
CLUSTER_INFO = {
    # Age ~54 | Income ~$36K | Spending ~$131 | Children ~1.2 | Living with Partner
    0: {
        "segment": "Budget-Conscious Families",
        "suggestion": (
            "Target with family bundle offers, loyalty cashback, and weekend flash sales. "
            "Highlight affordable everyday essentials and multi-buy discounts."
        ),
    },
    # Age ~62 | Income ~$60K | Spending ~$796 | Children ~1.1 | High web + store purchases
    1: {
        "segment": "Mature Mid-Range Shoppers",
        "suggestion": (
            "Cross-sell across web and in-store channels. "
            "Offer curated subscription boxes and loyalty tier upgrades to drive repeat purchases."
        ),
    },
    # Age ~58 | Income ~$75K | Spending ~$1,306 | Children ~0.3 | Top catalog + store buyers
    2: {
        "segment": "Affluent Premium Buyers",
        "suggestion": (
            "Engage with exclusive VIP early access, personalised luxury recommendations, "
            "and premium catalogue deals across wines, meats, and gold products."
        ),
    },
    # Age ~56 | Income ~$38K | Spending ~$183 | Children ~1.2 | Living Alone (single/divorced)
    3: {
        "segment": "Independent Deal Seekers",
        "suggestion": (
            "Re-engage with personalised solo-shopper deals and coupon campaigns. "
            "Promote app-exclusive discounts and convenience bundles for single-person households."
        ),
    },
}

# ---------------------------------------------------------------------------
# Load artifacts at startup (fail loudly if files are missing)
# ---------------------------------------------------------------------------
def _load(name: str):
    path = BASE_DIR / name
    if not path.exists():
        raise FileNotFoundError(
            f"Artifact '{name}' not found in {BASE_DIR}. "
            "Export it from SmartCart.ipynb first."
        )
    return joblib.load(path)


def _load_feature_cols() -> list:
    path = BASE_DIR / "feature_cols.json"
    if not path.exists():
        raise FileNotFoundError(
            "feature_cols.json not found. Export it from SmartCart.ipynb first."
        )
    with open(path) as fh:
        return json.load(fh)


try:
    ohe = _load("ohe.pkl")
    scaler = _load("scaler.pkl")
    pca = _load("pca.pkl")
    kmeans = _load("kmeans.pkl")
    feature_cols = _load_feature_cols()
    ARTIFACTS_READY = True
    ARTIFACTS_ERROR = None
except FileNotFoundError as _err:
    ARTIFACTS_READY = False
    ARTIFACTS_ERROR = str(_err)
    print(f"[WARN] {ARTIFACTS_ERROR}")

# ---------------------------------------------------------------------------
# Categorical column names used during OHE (must match training)
# ---------------------------------------------------------------------------
CAT_COLS = ["Education", "Living_With"]

# Numeric inputs (alignment is done via feature_cols)
NUMERIC_FIELDS = [
    "Age",
    "Income",
    "Total_children",
    "Total_spending",
    "Recency",
    "Customer_Tenure_Days",
    "NumDealsPurchases",
    "NumWebPurchases",
    "NumCatalogPurchases",
    "NumStorePurchases",
    "NumWebVisitsMonth",
    "Complain",
    "Response",
]


# ---------------------------------------------------------------------------
# Helper: build a one-row DataFrame aligned to feature_cols
# ---------------------------------------------------------------------------
def build_feature_row(form_data: dict) -> pd.DataFrame:
    """
    1. Parse numeric fields from form_data.
    2. OHE-encode Education and Living_With.
    3. Concatenate into a single-row DataFrame with columns = feature_cols.
    """
    # -- numeric part
    numeric_row = {field: float(form_data[field]) for field in NUMERIC_FIELDS}

    # -- OHE part
    cat_input = pd.DataFrame(
        [[form_data["Education"], form_data["Living_With"]]],
        columns=CAT_COLS,
    )
    enc_array = ohe.transform(cat_input)
    enc_df = pd.DataFrame(
        enc_array.toarray() if hasattr(enc_array, "toarray") else enc_array,
        columns=ohe.get_feature_names_out(CAT_COLS),
    )

    # -- merge into one row
    row = {**numeric_row, **enc_df.iloc[0].to_dict()}
    df = pd.DataFrame([row])

    # -- align to training column order (fills any missing cols with 0)
    df = df.reindex(columns=feature_cols, fill_value=0)
    return df


# ---------------------------------------------------------------------------
# Helper: generate a human-readable reason for the predicted cluster
# ---------------------------------------------------------------------------
# Cluster centroids (from ANOVA / cluster summary analysis)
_CENTROIDS = {
    "Total_spending":       {0: 131,   1: 796,   2: 1306,  3: 183},
    "Income":               {0: 35605, 1: 59857, 2: 74671, 3: 38097},
    "NumCatalogPurchases":  {0: 0.6,   1: 3.5,   2: 5.7,   3: 0.9},
    "NumStorePurchases":    {0: 3.5,   1: 7.6,   2: 8.6,   3: 3.8},
    "Total_children":       {0: 1.2,   1: 1.1,   2: 0.3,   3: 1.2},
    "Living_With":          {0: "Partner", 1: "Partner", 2: "Alone", 3: "Alone"},
    "Education":            {0: "Graduate", 1: "Graduate", 2: "Postgraduate", 3: "Graduate"},
}


def build_reason(data: dict, cluster_id: int) -> str:
    """Return a plain-English explanation of why the input landed in this cluster."""
    spending  = float(data.get("Total_spending", 0))
    income    = float(data.get("Income", 0))
    catalog   = float(data.get("NumCatalogPurchases", 0))
    store     = float(data.get("NumStorePurchases", 0))
    children  = float(data.get("Total_children", 0))
    living    = data.get("Living_With", "")
    education = data.get("Education", "")

    parts = []

    # --- Axis 1: Spending & Income (strongest separator, F≈1800 / 1216) ---
    if spending >= 900:
        parts.append(f"high total spending (${spending:,.0f}) signals a premium buyer")
    elif spending >= 400:
        parts.append(f"moderate total spending (${spending:,.0f}) places this customer in a mid-range tier")
    else:
        parts.append(f"low total spending (${spending:,.0f}) indicates a budget-conscious buyer")

    if income >= 65000:
        parts.append(f"high income (${income:,.0f}) further confirms an affluent profile")
    elif income >= 50000:
        parts.append(f"mid-range income (${income:,.0f}) aligns with moderate-spending clusters")
    else:
        parts.append(f"lower income (${income:,.0f}) is consistent with value-oriented segments")

    # --- Axis 2: Living situation (PC2 — key C0 vs C3 separator, F≈1003) ---
    if living == "Alone":
        parts.append("living alone (single/divorced) is a strong indicator for independent-shopper segments")
    else:
        parts.append("living with a partner is characteristic of family-oriented segments")

    # --- Axis 3: Purchase channels (F≈910 / 750) ---
    if catalog >= 4:
        parts.append(f"high catalogue purchases ({catalog:.0f}) suggest a premium, curated-shopping preference")
    elif store >= 7:
        parts.append(f"frequent in-store purchases ({store:.0f}) reflect an active, high-engagement shopper")

    # --- Axis 4: Children & Education (supplementary) ---
    if children == 0:
        parts.append("no children at home correlates with higher discretionary spending")
    elif children >= 2:
        parts.append(f"{children:.0f} children at home is typical of family-budget segments")

    if education == "Postgraduate":
        parts.append("postgraduate education is associated with higher-income, premium-spending clusters")

    # Join into a readable sentence
    if len(parts) == 1:
        return parts[0].capitalize() + "."
    head = parts[0].capitalize()
    tail = "; ".join(parts[1:])
    return f"{head}, and {tail}."


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if not ARTIFACTS_READY:
        return jsonify({"error": ARTIFACTS_ERROR}), 503

    try:
        # Accept both JSON and form-encoded bodies
        data = request.get_json(silent=True) or request.form.to_dict()

        # Build aligned feature row
        df = build_feature_row(data)

        # Pipeline: scale -> PCA -> cluster
        X_scaled = scaler.transform(df)
        X_pca = pca.transform(X_scaled)
        cluster_id = int(kmeans.predict(X_pca)[0])

        info = CLUSTER_INFO.get(
            cluster_id,
            {"segment": f"Cluster {cluster_id}", "suggestion": "No info available."},
        )

        reason = build_reason(data, cluster_id)

        return jsonify(
            {
                "cluster_id": cluster_id,
                "segment":    info["segment"],
                "suggestion": info["suggestion"],
                "reason":     reason,
            }
        )

    except KeyError as exc:
        return jsonify({"error": f"Missing field: {exc}"}), 400
    except ValueError as exc:
        return jsonify({"error": f"Invalid value: {exc}"}), 400
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True, port=5000)
