# Insurance Cost Predictor

A web application that predicts medical insurance costs using an XGBoost machine learning model with 21 engineered features.

## Quick Start (Local)

```bash
cd classic-insurance/render-app
pip install -r requirements.txt
python app.py
# Open http://localhost:5000
```

## Deploy to Render

1. Push this repo to GitHub
2. Go to [render.com](https://render.com) → **New** → **Web Service**
3. Connect your GitHub repo
4. Set:
   - **Root Directory**: `classic-insurance/render-app`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app --bind 0.0.0.0:$PORT`
5. Click **Create Web Service**

Or use the `render.yaml` blueprint for one-click deploy.

## Model Info

| Property | Value |
|----------|-------|
| Algorithm | XGBoost Regressor |
| Features | 21 engineered features |
| Target | `log1p(charges)` |
| Dataset | Medical Cost Personal (1,338 rows) |

## API

**POST** `/api/predict`

```json
{
    "age": 30,
    "sex": "male",
    "bmi": 25.0,
    "children": 0,
    "smoker": "no",
    "region": "northeast"
}
```

Response:
```json
{
    "predicted_charges": 4111.67,
    "features": { ... },
    "warnings": []
}
```
