"""
Smart Sales - Streamlit demo
This demo trains two models on synthetic sales data:
 - Regression model to predict next month's sales amount
 - Classification model to predict whether sales will increase next month (Up/Down)
Requirements:
    pip install streamlit pandas scikit-learn matplotlib numpy
Run:
    streamlit run sales_streamlit_demo.py
"""
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, classification_report
import matplotlib.pyplot as plt

st.set_page_config(page_title="Smart Sales", layout="centered")

st.title("Smart Sales")
st.caption("This app predicts next month's sales amount (regression) and whether sales will increase (classification).")
st.write("Synthetic data is used for demo purposes. Use your real dataset for production.")

@st.cache_data(show_spinner=False)
def generate_synthetic_sales_data(n_months=48, n_stores=5, random_state=42):
    np.random.seed(random_state)
    rows = []
    for store in range(1, n_stores+1):
        base = np.random.uniform(20000, 80000)  # base monthly sales for the store
        trend = np.random.uniform(-500, 1500)   # monthly trend
        season_amp = np.random.uniform(0.05, 0.25)  # seasonality amplitude
        for m in range(n_months):
            month_idx = m + 1
            month_of_year = (m % 12) + 1
            season = 1 + season_amp * np.sin(2 * np.pi * (month_of_year / 12))
            marketing = max(0, np.random.normal(5000, 2000))
            price = np.random.uniform(10, 50)
            competitor_price = price + np.random.normal(0, 3)
            holiday = 1 if month_of_year in [1,4,12] and np.random.rand() < 0.6 else 0
            noise = np.random.normal(0, base*0.05)
            current_sales = max(1000, base + trend*m) * season + marketing*0.5 - price*50 + noise + holiday*2000 - competitor_price*10
            rows.append({
                'store_id': f"Store_{store}",
                'month': month_idx,
                'month_of_year': month_of_year,
                'marketing_spend': round(marketing, 2),
                'price': round(price, 2),
                'competitor_price': round(competitor_price, 2),
                'holiday': holiday,
                'current_sales': round(current_sales, 2)
            })
    df = pd.DataFrame(rows)
    # Sort and compute next_month_sales per store
    df = df.sort_values(['store_id','month']).reset_index(drop=True)
    df['next_month_sales'] = df.groupby('store_id')['current_sales'].shift(-1)
    df = df.dropna().reset_index(drop=True)
    # create target for classification: sales increase next month?
    df['increase_flag'] = (df['next_month_sales'] > df['current_sales']).astype(int)
    return df

df = generate_synthetic_sales_data(n_months=36, n_stores=6)

@st.cache_resource(show_spinner=False)
def train_models(df):
    features = ['store_id','month_of_year','marketing_spend','price','competitor_price','holiday','current_sales']
    X = df[features].copy()
    y_reg = df['next_month_sales'].copy()
    y_clf = df['increase_flag'].copy()
    # preprocessing: one-hot store_id and month_of_year, scale numeric
    cat_cols = ['store_id','month_of_year']
    num_cols = ['marketing_spend','price','competitor_price','holiday','current_sales']
    preproc = ColumnTransformer([
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ])
    # regression pipeline
    reg_pipe = Pipeline([
        ('pre', preproc),
        ('reg', RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1))
    ])
    # classification pipeline
    clf_pipe = Pipeline([
        ('pre', preproc),
        ('clf', RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1))
    ])
    X_train, X_test, yreg_train, yreg_test, yclf_train, yclf_test = train_test_split(
        X, y_reg, y_clf, test_size=0.2, random_state=42, stratify=y_clf)
    reg_pipe.fit(X_train, yreg_train)
    clf_pipe.fit(X_train, yclf_train)
    # metrics
    yreg_pred = reg_pipe.predict(X_test)
    yclf_pred = clf_pipe.predict(X_test)
    reg_metrics = {'mae': mean_absolute_error(yreg_test, yreg_pred), 'r2': r2_score(yreg_test, yreg_pred)}
    clf_metrics = {'accuracy': accuracy_score(yclf_test, yclf_pred), 'report': classification_report(yclf_test, yclf_pred, output_dict=True)}
    return {'reg_model': reg_pipe, 'clf_model': clf_pipe, 'features': features, 'reg_metrics': reg_metrics, 'clf_metrics': clf_metrics, 'classes': clf_pipe.classes_}

state = train_models(df)

# Sidebar inputs for a single prediction
st.sidebar.header("Enter current month data (single record)")
store = st.sidebar.selectbox("Store", sorted(df['store_id'].unique()))
month_of_year = st.sidebar.selectbox("Month of Year", sorted(df['month_of_year'].unique()))
marketing_spend = st.sidebar.number_input("Marketing Spend", min_value=0.0, value=3000.0, step=100.0, format="%.2f")
price = st.sidebar.number_input("Price per unit", min_value=0.1, value=25.0, step=0.1, format="%.2f")
competitor_price = st.sidebar.number_input("Competitor Price", min_value=0.1, value=26.0, step=0.1, format="%.2f")
holiday = st.sidebar.selectbox("Holiday month (1=yes, 0=no)", [0,1])
current_sales = st.sidebar.number_input("Current Month Sales (amount)", min_value=0.0, value=45000.0, step=100.0, format="%.2f")

if st.sidebar.button("Predict"):
    rec = pd.DataFrame([{
        'store_id': store,
        'month_of_year': month_of_year,
        'marketing_spend': marketing_spend,
        'price': price,
        'competitor_price': competitor_price,
        'holiday': holiday,
        'current_sales': current_sales
    }])
    reg_model = state['reg_model']
    clf_model = state['clf_model']
    pred_sales = reg_model.predict(rec)[0]
    prob_increase = None
    if hasattr(clf_model, "predict_proba"):
        prob = clf_model.predict_proba(rec)[0]
        # classes order
        classes = clf_model.classes_
        prob_df = pd.DataFrame({'class': classes, 'probability': prob})
        prob_df = prob_df.sort_values('probability', ascending=False).reset_index(drop=True)
        prob_increase = prob_df
    increase_pred = clf_model.predict(rec)[0]
    st.subheader("Predictions for next month")
    st.success(f"Predicted next month sales: {pred_sales:,.2f}")
    st.write(f"Classification (increase next month?): {'Yes' if increase_pred==1 else 'No'}")
    if prob_increase is not None:
        st.write("Classification probabilities:")
        st.table(prob_increase)
        # plot probabilities
        fig, ax = plt.subplots()
        ax.bar(prob_increase['class'].astype(str), prob_increase['probability'])
        ax.set_ylim(0,1)
        ax.set_ylabel("Probability")
        ax.set_title("Probability of classes (increase=1, decrease=0)")
        st.pyplot(fig)
    # simple sensitivity: show how predicted sales change with +/-10% marketing
    adj = rec.copy()
    adj['marketing_spend_low'] = adj['marketing_spend'] * 0.9
    adj['marketing_spend_high'] = adj['marketing_spend'] * 1.1
    rec_low = rec.copy(); rec_low['marketing_spend'] = adj['marketing_spend_low']
    rec_high = rec.copy(); rec_high['marketing_spend'] = adj['marketing_spend_high']
    low_pred = reg_model.predict(rec_low)[0]
    high_pred = reg_model.predict(rec_high)[0]
    st.write("Quick sensitivity: effect of ±10% marketing spend")
    st.write(f"-10% marketing => predicted sales: {low_pred:,.2f}")
    st.write(f"+10% marketing => predicted sales: {high_pred:,.2f}")

# Show dataset and model metrics
st.markdown("---")
st.subheader("Sample synthetic sales data")
st.dataframe(df.sample(8, random_state=2).reset_index(drop=True))

st.subheader("Regression model metrics (test set)")
st.write(f"MAE: {state['reg_metrics']['mae']:.2f}")
st.write(f"R²: {state['reg_metrics']['r2']:.3f}")

st.subheader("Classification model metrics (test set)")
st.write(f"Accuracy: {state['clf_metrics']['accuracy']:.3f}")
report_df = pd.DataFrame(state['clf_metrics']['report']).transpose()
st.dataframe(report_df)

st.markdown("---")
st.write("Notes: This demo uses synthetic data and baseline Random Forest models. Replace data with real sales data for production and consider hyperparameter tuning and feature engineering for better results.")