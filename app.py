from pathlib import Path
import joblib
import pandas as pd
from flask import Flask, render_template, request, send_file, redirect, url_for

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"


# ============================================================
# LOAD MODELS
# ============================================================

try:
    encoder = joblib.load(MODEL_DIR / "encoder.pkl")
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    pca = joblib.load(MODEL_DIR / "pca.pkl")
    kmeans = joblib.load(MODEL_DIR / "kmeans.pkl")
    cluster_profiles = joblib.load(
        MODEL_DIR / "cluster_profiles.pkl"
    )
    feature_cols = joblib.load(
        MODEL_DIR / "feature_cols.pkl"
    )

    try:
        metadata = joblib.load(
            MODEL_DIR / "metadata.pkl"
        )
    except Exception:
        metadata = {}

    MODELS_READY = True
    MODEL_ERROR = ""

except Exception as e:
    MODELS_READY = False
    MODEL_ERROR = str(e)


# ============================================================
# FEATURE ENGINEERING
# ============================================================

REQUIRED_COLUMNS = [
    "Income",
    "Year_Birth",
    "Dt_Customer",
    "Marital_Status",
    "Kidhome",
    "Teenhome",
    "MntWines",
    "MntFruits",
    "MntMeatProducts",
    "MntFishProducts",
    "MntSweetProducts",
    "MntGoldProds",
    "Education"
]

def engineer_data(df, use_current_date=False):

    df = df.copy()

    if "Marital_Status" not in df.columns:
        df["Marital_Status"] = "Single"

    if "Education" not in df.columns:
        df["Education"] = "Graduation"

    missing = [
        c for c in REQUIRED_COLUMNS
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing columns: " + ", ".join(missing)
        )

    # Income
    df["Income"] = pd.to_numeric(
        df["Income"],
        errors="coerce"
    )

    df["Income"] = df["Income"].fillna(
        df["Income"].median()
    )

    # Date
    dates = pd.to_datetime(
        df["Dt_Customer"],
        dayfirst=True,
        errors="coerce"
    )

    if dates.isna().any():
        raise ValueError(
            "Some Dt_Customer values are invalid."
        )
    
    if use_current_date:
        reference_year = pd.Timestamp.now().year
        reference_date = pd.Timestamp.now().normalize()
    else:
        reference_year = int(
            metadata.get(
                "reference_year",
                dates.dt.year.max()
            )
        )

        reference_date = pd.Timestamp(
            metadata.get(
                "reference_date",
                dates.max().date()
            )
        )

    # Age
    df["Age"] = (
        reference_year -
        pd.to_numeric(
            df["Year_Birth"],
            errors="coerce"
        )
    ).clip(lower=0)

    # Customer tenure
    df["Customer_Tenure_Days"] = (
        reference_date - dates
    ).dt.days

    # Spending
    spending = [
        "MntWines",
        "MntFruits",
        "MntMeatProducts",
        "MntFishProducts",
        "MntSweetProducts",
        "MntGoldProds"
    ]

    for col in spending + ["Kidhome", "Teenhome"]:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        ).fillna(0)

    df["Total_Spending"] = df[spending].sum(axis=1)

    df["Total_Children"] = (
        df["Kidhome"] + df["Teenhome"]
    )

    # Education
    df["Education"] = df["Education"].fillna("Graduation").replace({
        "Basic": "Undergraduate",
        "2n Cycle": "Undergraduate",
        "Graduation": "Graduate",
        "Master": "Postgraduate",
        "PhD": "Postgraduate"
    })

    # Living arrangement
    df["Living_With"] = df["Marital_Status"].fillna("Single").replace({
        "Married": "Partner",
        "Together": "Partner",
        "Single": "Alone",
        "Divorced": "Alone",
        "Widow": "Alone",
        "Absurd": "Alone",
        "YOLO": "Alone"
    })

    # Remove outliers
    before = len(df)

    df = df[
        (df["Age"] < 90) &
        (df["Income"] < 600000)
    ].copy()

    removed = before - len(df)

    return df, removed

# ============================================================
# PREPARE DATA FOR MODEL
# ============================================================

def prepare_for_model(df):

    encoded = encoder.transform(
        df[["Education", "Living_With"]]
    )

    encoded_df = pd.DataFrame(
        encoded,
        columns=encoder.get_feature_names_out(
            ["Education", "Living_With"]
        ),
        index=df.index
    )

    numeric_df = df.drop(
        columns=[
            "Education",
            "Living_With",
            "Response"
        ],
        errors="ignore"
    )

    X = pd.concat(
        [numeric_df, encoded_df],
        axis=1
    )

    X = X.reindex(
        columns=feature_cols,
        fill_value=0
    )

    return X


# ============================================================
# PREDICTION
# ============================================================

def predict(df):

    X = prepare_for_model(df)

    X_scaled = scaler.transform(X)

    X_pca = pca.transform(X_scaled)

    predictions = kmeans.predict(X_pca)

    return predictions, X_pca


# ============================================================
# CLUSTER LABEL
# ============================================================

def cluster_label(cluster):

    try:

        profile = cluster_profiles.loc[cluster]

        spending = profile.get(
            "Total_Spending",
            0
        )

        income = profile.get(
            "Income",
            0
        )

        median_spending = cluster_profiles[
            "Total_Spending"
        ].median()

        median_income = cluster_profiles[
            "Income"
        ].median()

        if spending > median_spending * 1.5:
            return "High Value Customer"

        if spending > median_spending:
            return "Active Spender"

        if income > median_income * 1.3:
            return "High Income / Lower Spend"

        return "General Customer"

    except Exception:
        return f"Customer Segment {cluster}"


# ============================================================
# DASHBOARD DATA
# ============================================================

def dashboard_data(df, uploaded=False):

    predictions, X_pca = predict(df)

    result_df = df.copy()

    result_df["Cluster"] = predictions

    clusters = sorted(
        result_df["Cluster"].unique()
    )

    profiles = []

    for cluster in clusters:

        group = result_df[
            result_df["Cluster"] == cluster
        ]

        profiles.append({

            "cluster": int(cluster),

            "label": cluster_label(
                cluster
            ),

            "customers": len(group),

            "share": round(
                len(group) /
                len(result_df) * 100,
                1
            ),

            "avg_income": round(
                group["Income"].mean(),
                2
            ),

            "avg_spending": round(
                group["Total_Spending"].mean(),
                2
            ),

            "avg_age": round(
                group["Age"].mean(),
                1
            ),

            "avg_children": round(
                group["Total_Children"].mean(),
                1
            )
        })

    points = []

    for i in range(len(result_df)):

        points.append({

            "x": float(
                X_pca[i][0]
            ),

            "y": float(
                X_pca[i][1]
            ),

            "cluster": int(
                predictions[i]
            )
        })

    return {

        "uploaded": uploaded,

        "total_customers": len(
            result_df
        ),

        "num_clusters": len(
            clusters
        ),

        "avg_spending": round(
            result_df[
                "Total_Spending"
            ].mean(),
            2
        ),

        "avg_income": round(
            result_df[
                "Income"
            ].mean(),
            2
        ),

        "cluster_ids": [int(c) for c in clusters],

        "cluster_counts": [
            int(
                (predictions == c).sum()
            )
            for c in clusters
        ],

        "profiles": profiles,

        "pca_points": points,

        "pca_variance": [
            round(
                float(x) * 100,
                2
            )
            for x in
            pca.explained_variance_ratio_
        ]
    }


# ============================================================
# DASHBOARD
# ============================================================

@app.route("/home")
def home():
    return render_template("index.html", data={"uploaded": False})

@app.route("/")
def index():

    if not MODELS_READY:

        return render_template(
            "error.html",
            error=
            "Model files could not be loaded: "
            + MODEL_ERROR
        )

    try:

        uploaded_file = (
            BASE_DIR /
            "uploaded_customer_data.csv"
        )

        # ----------------------------------------------------
        # USER HAS UPLOADED A CSV
        # ----------------------------------------------------

        if uploaded_file.exists():

            df = pd.read_csv(
                uploaded_file
            )

            df, removed = engineer_data(
                df
            )

            data = dashboard_data(
                df,
                uploaded=True
            )

        # ----------------------------------------------------
        # DEFAULT DATASET
        # ----------------------------------------------------

        else:

            df = pd.read_csv(
                BASE_DIR /
                "smartcart_customers.csv"
            )

            df, removed = engineer_data(
                df
            )

            data = dashboard_data(
                df
            )

        data["removed_rows"] = removed

        return render_template(
            "index.html",
            data=data
        )

    except Exception as e:

        print(
            "DASHBOARD ERROR:",
            repr(e)
        )

        return render_template(
            "error.html",
            error=str(e)
        )


# ============================================================
# INDIVIDUAL CUSTOMER
# ============================================================

@app.route(
    "/analyze",
    methods=["GET", "POST"]
)
def analyze():

    if request.method == "GET":

        return render_template(
            "analyze.html"
        )

    try:

        customer = {

            "Income": float(
                request.form["Income"]
            ),

            "Year_Birth": int(
                request.form["Year_Birth"]
            ),

            "Dt_Customer":
                request.form["Dt_Customer"],

            "Marital_Status":
                request.form["Marital_Status"],

            "Kidhome": int(
                request.form["Kidhome"]
            ),

            "Teenhome": int(
                request.form["Teenhome"]
            ),

            "MntWines": float(
                request.form["MntWines"]
            ),

            "MntFruits": float(
                request.form["MntFruits"]
            ),

            "MntMeatProducts": float(
                request.form[
                    "MntMeatProducts"
                ]
            ),

            "MntFishProducts": float(
                request.form[
                    "MntFishProducts"
                ]
            ),

            "MntSweetProducts": float(
                request.form[
                    "MntSweetProducts"
                ]
            ),

            "MntGoldProds": float(
                request.form[
                    "MntGoldProds"
                ]
            ),

            "Education":
                request.form["Education"]
        }

        df = pd.DataFrame(
            [customer]
        )

        df, _ = engineer_data(
            df,
            use_current_date=True
        )

        predictions, X_pca = predict(
            df
        )

        cluster = int(
            predictions[0]
        )

        result = {

            "cluster": cluster,

            "label": cluster_label(
                cluster
            ),

            "pca1": round(
                float(
                    X_pca[0][0]
                ),
                3
            ),

            "pca2": round(
                float(
                    X_pca[0][1]
                ),
                3
            ),

            "income": round(
                float(
                    df.iloc[0][
                        "Income"
                    ]
                ),
                2
            ),

            "spending": round(
                float(
                    df.iloc[0][
                        "Total_Spending"
                    ]
                ),
                2
            ),

            "age": int(
                df.iloc[0]["Age"]
            )
        }

        return render_template(
            "result.html",
            result=result
        )

    except Exception as e:

        print(
            "ANALYZE ERROR:",
            repr(e)
        )

        return render_template(
            "error.html",
            error=str(e)
        )

# ============================================================
# CSV UPLOAD
# ============================================================

@app.route(
    "/upload",
    methods=["GET", "POST"]
)
def upload():

    if request.method == "GET":

        return render_template(
            "upload.html"
        )

    file = request.files.get(
        "file"
    )

    if not file or file.filename == "":

        return render_template(
            "error.html",
            error=
            "Please choose a CSV file."
        )

    if not file.filename.lower().endswith(
        ".csv"
    ):

        return render_template(
            "error.html",
            error=
            "Please upload a CSV file."
        )

    try:

        # ====================================================
        # READ ORIGINAL CSV
        # ====================================================

        raw_df = pd.read_csv(
            file
        )

        # ====================================================
        # ENGINEER A COPY FOR ML
        # ====================================================

        df, removed = engineer_data(
            raw_df
        )

        # ====================================================
        # PREDICT CLUSTERS
        # ====================================================

        predictions, X_pca = predict(
            df
        )

        # ====================================================
        # SAVE SEGMENTED CSV
        # ====================================================

        segmented_df = df.copy()

        segmented_df["Cluster"] = (
            predictions
        )

        segmented_df.to_csv(
            BASE_DIR /
            "smartcart_segmented.csv",
            index=False
        )

        # ====================================================
        # SAVE ORIGINAL RAW CSV
        #
        # IMPORTANT:
        # Dashboard will engineer this ONCE.
        # ====================================================

        raw_df.to_csv(
            BASE_DIR /
            "uploaded_customer_data.csv",
            index=False
        )

        return redirect(
            url_for("index")
        )

    except Exception as e:

        print(
            "UPLOAD ERROR:",
            repr(e)
        )

        return render_template(
            "error.html",
            error=
            f"Could not process CSV: {e}"
        )


# ============================================================
# DOWNLOAD
# ============================================================

@app.route(
    "/download-segmented"
)
def download_segmented():

    path = (
        BASE_DIR /
        "smartcart_segmented.csv"
    )

    if not path.exists():

        return render_template(
            "error.html",
            error=
            "No segmented CSV available yet."
        )

    return send_file(
        path,
        as_attachment=True,
        download_name=
        "smartcart_segmented.csv"
    )


# ============================================================
# RESET
# ============================================================

@app.route("/reset")
def reset():

    for filename in [

        "uploaded_customer_data.csv",

        "smartcart_segmented.csv"

    ]:

        path = (
            BASE_DIR /
            filename
        )

        if path.exists():
            path.unlink()

    return redirect(
        url_for("index")
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/health")
def health():

    return {

        "status": "ok",

        "models_ready":
            MODELS_READY

    }


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    app.run(debug=False)