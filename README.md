# SmartCart

## E-Commerce Customer Segmentation

### Project Overview
SmartCart is an unsupervised machine learning application that segments e-commerce consumers based on their purchasing patterns, demographics, channel preferences, and campaign responses. By applying Principal Component Analysis (PCA) and Agglomerative Clustering to customer data, SmartCart transforms complex multi-dimensional datasets into 4 distinct, actionable customer personas.

### Problem Statement
E-commerce platforms collect large volumes of customer demographic and purchase records. However, raw data alone does not explain customer purchasing behavior. Generic, one-size-fits-all marketing campaigns treat all consumers identically, leading to low conversion rates, high customer acquisition costs, and inefficient promotional spending. Retailers require an automated machine learning approach to group customers into meaningful behavioral segments.

### Objective
The primary objective of SmartCart is to develop a complete data science and machine learning pipeline that:
1. Cleans and preprocesses consumer transactional and demographic data.
2. Engineers meaningful household and spending metrics.
3. Standardizes features and reduces dimensionality via 3D Principal Component Analysis (PCA).
4. Determines the optimal cluster count $K=4$ using KneeLocator Elbow analysis and Silhouette Evaluation.
5. Performs Agglomerative Hierarchical Clustering to define 4 customer personas.
6. Provides an interactive web application (Flask) for real-time customer segmentation inference.

### Dataset
The project utilizes the `smartcart_customers.csv` dataset, which contains **2,240 raw customer records** and **22 columns**:
- **Demographic Information**: `Year_Birth`, `Education`, `Marital_Status`, `Income`, `Kidhome`, `Teenhome`.
- **Product Category Spending**: `MntWines`, `MntFruits`, `MntMeatProducts`, `MntFishProducts`, `MntSweetProducts`, `MntGoldProds`.
- **Purchase Channels & Engagement**: `NumWebPurchases`, `NumCatalogPurchases`, `NumStorePurchases`, `NumDealsPurchases`, `NumWebVisitsMonth`, `Recency`, `Complain`, `Response`.

### Methodology
1. **Data Preprocessing & Outlier Handling**:
   - Imputed 24 missing values in `Income` using median imputation.
   - Removed extreme outliers (`Age >= 90` and `Income >= $600,000`), yielding 2,236 clean customer samples.
2. **Feature Engineering**:
   - Calculated `Age = 2026 - Year_Birth`.
   - Calculated `Customer_Tenure_Days` relative to maximum customer join date.
   - Summed total category spending into `Total_spending`.
   - Combined `Kidhome` and `Teenhome` into `Total_children`.
   - Grouped `Education` into 3 tiers (`Graduate`, `Postgraduate`, `Undergraduate`) and `Marital_Status` into `Living_With` (`Partner` vs `Alone`).
3. **Encoding & Scaling**:
   - Applied `OneHotEncoder` on categorical variables generating 5 binary dummy features.
   - Applied `StandardScaler` across all 18 numerical and encoded model features.
4. **Dimensionality Reduction & Evaluation**:
   - Fitted `PCA(n_components=3)` capturing 44.96% cumulative explained variance.
   - Used `KneeLocator` on WCSS inertia to establish $K=4$ elbow point and validated cluster separation using Silhouette Scores.
5. **Clustering & Model Persistence**:
   - Fitted `AgglomerativeClustering(n_clusters=4, linkage='ward')` to compute cluster personas and saved model artifacts (`scaler`, `pca`, `centroids`) to `model.pkl`.

### Machine Learning Algorithms
- **StandardScaler**: Feature standardization ($\mu=0, \sigma=1$).
- **OneHotEncoder**: Categorical variable dummy encoding.
- **PCA (Principal Component Analysis)**: 3D linear dimensionality reduction.
- **Elbow Method (KneeLocator)**: WCSS inertia curve elbow detection for optimal $K$.
- **Silhouette Analysis**: Cluster cohesion and separation validation metric.
- **Agglomerative Clustering**: Hierarchical bottom-up clustering using Ward linkage.
- **K-Means Clustering**: Centroid distance initialization and comparative clustering.

### Features
- **Exploratory ML Analysis Dashboard**: Preprocessing notes, $K=4$ elbow analysis, PCA reduction metrics, and cluster characterization table.
- **Interactive Segmentation Inference Engine**: Live customer input form running real-time preprocessing, scaling, 3D PCA vector projection, and nearest centroid cluster assignment.
- **Customer Personas**:
  - **Cluster 0**: *Budget Partnered Families* (Avg Income: $39,681 | Avg Spending: $222 | 100% Partnered)
  - **Cluster 1**: *Affluent Partnered High-Spenders* (Avg Income: $72,808 | Avg Spending: $1,237 | 100% Partnered)
  - **Cluster 2**: *Budget Single Individuals* (Avg Income: $36,960 | Avg Spending: $166 | 99.3% Alone)
  - **Cluster 3**: *Affluent Single High-Responders* (Avg Income: $70,723 | Avg Spending: $1,190 | 100% Alone | 32% Response)

### Technologies Used
- **Programming Language**: Python 3.13
- **Data Manipulation**: Pandas, NumPy
- **Machine Learning**: Scikit-Learn
- **K Determination**: KneeLocator (`kneed`)
- **Data Visualization**: Matplotlib, Seaborn
- **Backend Framework**: Flask 3.1
- **Frontend**: HTML5, Vanilla CSS3

### Project Structure
```text
SmartCart/
│
├── SmartCart.ipynb           # Jupyter Notebook (ML Pipeline, EDA & Evaluation)
├── smartcart_customers.csv   # Customer Dataset
├── train_and_save.py         # Pipeline Trainer & Model Serializer
├── model.pkl                 # Saved Model Artifacts
├── app.py                    # Flask Web Backend
├── requirements.txt          # Dependencies
├── .gitignore                # Git Ignore Rules
└── README.md                 # Documentation
│
├── templates/                # HTML Templates
│   ├── base.html             # Master Layout & Header Navbar
│   ├── index.html            # Home Overview & Pipeline Stepper
│   ├── analysis.html         # ML Analysis & Cluster Characteristics
│   └── segmentation.html     # Interactive Segmentation Live Demo
│
└── static/                   # Web Assets
    ├── css/
    │   └── style.css         # Modern Dark CSS Design System
    ├── js/
    │   └── script.js         # Interactive UI Controller
    └── images/               # Image Directory
```

### How to Run

```bash
pip install -r requirements.txt
python app.py
```


