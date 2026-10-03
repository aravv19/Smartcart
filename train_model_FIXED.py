import os
import joblib
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from kneed import KneeLocator

DATA_FILE = "smartcart_customers.csv"
MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)

print("Loading dataset...")
df = pd.read_csv(DATA_FILE)

# -----------------------------
# Same preprocessing as notebook
# -----------------------------
df["Income"] = df["Income"].fillna(df["Income"].median())

reference_year = pd.to_datetime(df["Dt_Customer"], dayfirst=True).dt.year.max()
df["Age"] = (reference_year - df["Year_Birth"]).clip(lower=0)

df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], dayfirst=True)
reference_date = df["Dt_Customer"].max()
df["Customer_Tenure_Days"] = (reference_date - df["Dt_Customer"]).dt.days

df["Total_Spending"] = (
    df["MntWines"]
    + df["MntFruits"]
    + df["MntMeatProducts"]
    + df["MntFishProducts"]
    + df["MntSweetProducts"]
    + df["MntGoldProds"]
)

df["Total_Children"] = df["Kidhome"] + df["Teenhome"]

df["Education"] = df["Education"].replace({
    "Basic": "Undergraduate",
    "2n Cycle": "Undergraduate",
    "Graduation": "Graduate",
    "Master": "Postgraduate",
    "PhD": "Postgraduate"
})

df["Living_With"] = df["Marital_Status"].replace({
    "Married": "Partner",
    "Together": "Partner",
    "Single": "Alone",
    "Divorced": "Alone",
    "Widow": "Alone",
    "Absurd": "Alone",
    "YOLO": "Alone"
})

drop_cols = [
    "ID", "Year_Birth", "Marital_Status",
    "Kidhome", "Teenhome", "Dt_Customer",
    "MntWines", "MntFruits", "MntMeatProducts",
    "MntFishProducts", "MntSweetProducts", "MntGoldProds"
]

df_cleaned = df.drop(columns=drop_cols)

df_cleaned = df_cleaned[df_cleaned["Age"] < 90]
df_cleaned = df_cleaned[df_cleaned["Income"] < 600_000]

# -----------------------------
# Encode categorical variables
# -----------------------------
cat_cols = ["Education", "Living_With"]

ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
encoded = ohe.fit_transform(df_cleaned[cat_cols])

encoded_cols = list(ohe.get_feature_names_out(cat_cols))
enc_df = pd.DataFrame(
    encoded,
    columns=encoded_cols,
    index=df_cleaned.index
)

# IMPORTANT FIX:
# Response is a campaign outcome, so it is NOT used for clustering.
# Education and Living_With are replaced by their one-hot encoded columns.
X = pd.concat(
    [df_cleaned.drop(columns=cat_cols + ["Response"]), enc_df],
    axis=1
)

# Save the exact final feature order used by the trained model.
feature_cols = X.columns.tolist()
X = X[feature_cols]

# -----------------------------
# Scaling + PCA
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

# -----------------------------
# Find K
# -----------------------------
wcss = []
silhouette_scores = []
k_values = range(2, 11)

for k in range(1, 11):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_pca)
    wcss.append(km.inertia_)

for k in k_values:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_pca)
    silhouette_scores.append(silhouette_score(X_pca, labels))

knee = KneeLocator(
    range(1, 11),
    wcss,
    curve="convex",
    direction="decreasing"
)

optimal_k = knee.elbow

if optimal_k is None:
    optimal_k = list(k_values)[silhouette_scores.index(max(silhouette_scores))]

final_k = int(optimal_k)

print("Selected K:", final_k)

# -----------------------------
# Final K-Means
# -----------------------------
kmeans = KMeans(
    n_clusters=final_k,
    random_state=42,
    n_init=10
)

labels = kmeans.fit_predict(X_pca)

X_profile = X.copy()
X_profile["cluster"] = labels
cluster_summary = X_profile.groupby("cluster").mean(numeric_only=True)

# -----------------------------
# Save artifacts
# -----------------------------
joblib.dump(ohe, f"{MODEL_DIR}/encoder.pkl")
joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")
joblib.dump(pca, f"{MODEL_DIR}/pca.pkl")
joblib.dump(kmeans, f"{MODEL_DIR}/kmeans.pkl")
joblib.dump(cluster_summary, f"{MODEL_DIR}/cluster_profiles.pkl")
joblib.dump(feature_cols, f"{MODEL_DIR}/feature_cols.pkl")

metadata = {
    "optimal_k": final_k,
    "reference_year": int(reference_year),
    "reference_date": str(reference_date.date()),
    "pca_explained_variance": pca.explained_variance_ratio_.tolist(),
    "training_rows": int(len(df_cleaned)),
    "silhouette_scores": {
        str(k): float(score)
        for k, score in zip(k_values, silhouette_scores)
    }
}

joblib.dump(metadata, f"{MODEL_DIR}/metadata.pkl")

# -----------------------------
# Verification
# -----------------------------
sample_scaled = scaler.transform(X[feature_cols].head(5))
sample_pca = pca.transform(sample_scaled)
sample_prediction = kmeans.predict(sample_pca)

print("Sample predictions:", sample_prediction.tolist())
print("Number of clusters:", kmeans.n_clusters)
print("Saved model files in ./models/")
