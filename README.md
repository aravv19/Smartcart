# SmartCart — Customer Segmentation & Analytics Platform

SmartCart segments customers using PCA and K-Means clustering. Upload
a customer CSV and it groups customers into behavioural segments —
income, spending, age, household — then shows each group's profile
through an interactive dashboard. It can also classify a single
customer on the spot from a short form, without needing a CSV.

## Features

- **Batch upload** — upload a customer CSV, get back a segmented
  dashboard (cluster sizes, spending by segment, a PCA scatter map)
  plus a downloadable `smartcart_segmented.csv` with cluster
  assignments added.
- **Single customer check** — enter one customer's details directly
  and see which segment they fall into, without preparing a CSV.
- **Resilient CSV handling** — required columns (income, spend,
  dates) fail with a specific error message; optional columns
  (education, marital status) fall back to sensible defaults instead
  of rejecting the whole file.

## Folder structure

```text
SmartCart/
├── app.py
├── train_model.py
├── requirements.txt
├── Procfile
├── smartcart_customers.csv
├── smartcart_updated.ipynb
├── models/
├── templates/
│   ├── base.html
│   ├── index.html
│   ├── analyze.html
│   ├── result.html
│   ├── upload.html
│   └── error.html
└── static/
    └── style.css
```

## 1. Install packages

```bash
pip install -r requirements.txt
```

## 2. Train/save the model

Make sure `smartcart_customers.csv` is in the same folder as `train_model.py`.

```bash
python train_model.py
```

This creates the files inside `models/`.

## 3. Start Flask

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

## Important

The Flask app does NOT retrain the model. It loads the saved encoder,
scaler, PCA and K-Means model and uses them for inference.

The notebook remains the main data-science analysis. `train_model.py`
recreates the preprocessing/model-training flow so the web app can be
started without manually running every notebook cell.

## CSV upload

The Batch Upload page accepts the original SmartCart customer CSV and
returns `smartcart_segmented.csv` with cluster/segment assignments.

## Deployment

The included `Procfile` is ready for platforms that support Gunicorn.

The trained model artifacts in `models/*.pkl` are committed to this
repo, so the deployed app loads them directly at startup — there is
no retraining step on deploy. If you retrain locally with
`train_model.py`, re-commit the updated `.pkl` files for the
deployed version to pick up the changes.