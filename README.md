# Smart Sales - Improved Demo

This Streamlit demo contains two tuned Random Forest models trained on improved synthetic sales data:
- Regression: predicts next month's sales amount.
- Classification: predicts whether sales will increase next month (Yes/No).

Feature engineering added: lag, 3-month rolling mean, percent change.

## Installation

```bash
pip install -r requirements.txt
```

## Run locally

```bash
streamlit run sales_streamlit_improved.py
```

Open the local URL shown in the terminal (usually http://localhost:8501).

## Deploy to Streamlit Cloud

1. Push the files (`sales_streamlit_improved.py` and `requirements.txt`) to a GitHub repository.
2. Go to https://share.streamlit.io and connect your repository.
3. Deploy the app.

## Usage

- Use the left sidebar to enter current month features.
- Click **Predict** to see the regression prediction for next month's sales and the classification result (increase or not).
- Sensitivity check shows how ±10% marketing spend affects predicted sales.

---

This project uses synthetic data for demonstration only. For production use your real sales dataset and retrain the models.
