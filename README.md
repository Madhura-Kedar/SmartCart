# SmartCart — Customer Segmentation System

> **Unsupervised ML minor project** — KMeans clustering on e-commerce customer data,
> served through a Flask web app with a clean UI.

---

## Demo



Enter customer attributes → the model runs **OHE → StandardScaler → PCA (3D) → KMeans (k=4)**
and returns a segment name, a human-readable reason, and a tailored marketing tip.

---

## Segments

| Cluster | Segment | Avg Income | Avg Spending | Key trait |
|---------|---------|-----------|-------------|-----------|
| C0 | Budget-Conscious Families | $36K | $131 | Partner, 2 kids |
| C1 | Mature Mid-Range Shoppers | $60K | $796 | High web + store purchases |
| C2 | Affluent Premium Buyers | $75K | $1,306 | High catalog, no kids, Postgrad |
| C3 | Independent Deal Seekers | $38K | $183 | Living alone |

---

## Project Structure

```
SmartCart/
├── app.py                  # Flask app (GET / and POST /predict)
├── SmartCart.ipynb         # Full ML pipeline notebook
├── smartcart_customers.csv # Dataset (2,240 customers, 22 features)
├── templates/
│   └── index.html          # Frontend UI
├── ohe.pkl                 # Fitted OneHotEncoder
├── scaler.pkl              # Fitted StandardScaler
├── pca.pkl                 # Fitted PCA (3 components)
├── kmeans.pkl              # Fitted KMeans (k=4)
├── feature_cols.json       # Ordered feature column list
└── requirements.txt
```

---

## ML Pipeline

```
Raw CSV
  ↓ Fill missing Income (median)
  ↓ Feature engineering: Age, Customer_Tenure_Days, Total_spending, Total_children
  ↓ Simplify Education → Undergraduate / Graduate / Postgraduate
  ↓ Simplify Marital_Status → Living_With (Alone / Partner)
  ↓ Remove outliers (Age > 90, Income > 600K)
  ↓ OneHotEncoder (Education, Living_With)
  ↓ StandardScaler
  ↓ PCA (3 components — 45% variance explained)
  ↓ KMeans (k=4, random_state=42)
```

### Top Features by ANOVA F-statistic

| Feature | F-stat | Impact |
|---------|--------|--------|
| Total_spending | 1793 | Dominant |
| Income | 1216 | Dominant |
| Living_With | 1003 | Dominant |
| NumCatalogPurchases | 910 | Very High |
| NumStorePurchases | 750 | Very High |
| Recency | 0.5 | None |
| Complain | 0.5 | None |

---

## Setup & Run

```bash
# 1. Clone
git clone https://github.com/Madhura-Kedar/SmartCart.git
cd SmartCart

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the Flask app
python app.py
```



> **Note:** The `.pkl` files are included in the repo.
> To regenerate them, run all cells in `SmartCart.ipynb`.

---

## Dataset

**Marketing Campaign Dataset** — 2,240 customers with purchase history, demographics,
and campaign response data.

| Column | Description |
|--------|-------------|
| Income | Annual household income |
| MntWines / MntFruits / … | Amount spent per category |
| NumWebPurchases / NumStorePurchases / … | Purchase channel counts |
| Recency | Days since last purchase |
| Education | Highest education level |
| Marital_Status | Relationship status |

---

## Tech Stack

- **Python 3.x**
- **scikit-learn** — OHE, StandardScaler, PCA, KMeans, AgglomerativeClustering
- **Flask** — REST API backend
- **Pandas / NumPy** — data processing
- **Seaborn / Matplotlib** — EDA visualisations
- **HTML + CSS + Vanilla JS** — frontend UI

---

*Minor Project — Unsupervised Machine Learning*
