# SmartCart — Customer Segmentation & Analytics Platform

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
Do not commit the generated `.pkl` model files if you intend to train
them during deployment; otherwise remove their ignore rule and commit
the trained artifacts.
