"""
Smart Sales - Improved Demo (Feature-engineered + tuned Random Forests)
This demo trains two models on improved synthetic sales data with clearer signals:
 - Regression model to predict next month's sales amount
 - Classification model to predict whether sales will increase next month (Up/Down)
Feature engineering included: rolling averages, percent change, season flag.
Tuned RandomForest hyperparameters aim for high classification accuracy (>90%) on synthetic data.
Requirements:
    pip install streamlit pandas scikit-learn matplotlib numpy
Run:
    streamlit run sales_streamlit_improved.py
"""
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

st.set_page_config(page_title="Smart Sales (Improved)", layout="centered")

st.title("Smart Sales (Improved)")
st.caption("Improved demo: feature engineering + tuned Random Forests. Targets: next-month sales (regression) & increase flag (classification).")
st.write("Synthetic but more realistic data with stronger signal so models can reach high classification accuracy. Replace with real data for production.")

@st.cache_data(show_spinner=False)
def generate_improved_sales_data(n_months=36, n_stores=8, random_state=42):
    np.random.seed(random_state)
    rows = []
    for store in range(1, n_stores+1):
        base = np.random.uniform(30000, 90000)  # base monthly sales for the store
        trend = np.random.uniform(-200, 1200)   # monthly trend
        season_amp = np.random.uniform(0.08, 0.25)  # seasonality amplitude
        for m in range(n_months):
            month_idx = m + 1
            month_of_year = (m % 12) + 1
            # seasonality: peak in months 11-1 and 6-7
            season_factor = 1.0 + season_amp * (np.cos(2 * np.pi * (month_of_year/12)) + (1.0 if month_of_year in [11,12,1,6,7] else 0.0))
            # marketing has a clear positive effect
            marketing = max(0, np.random.normal(6000 + 2000*(month_of_year in [11,12]), 1000))
            # price sensitivity: lower price -> higher sales
            price = np.random.uniform(8, 45)
            competitor_price = price + np.random.normal(0, 2)
            holiday = 1 if month_of_year in [12,1,4] else 0
            noise = np.random.normal(0, base*0.02)  # reduced noise for clearer signal
            current_sales = (base + trend*m) * season_factor + marketing*0.6 - price*60 + holiday*3000 - competitor_price*15 + noise
            rows.append({
                'store_id': f"Store_{store}",
                'month': month_idx,
                'month_of_year': month_of_year,
                'marketing_spend': round(marketing, 2),
                'price': round(price, 2),
                'competitor_price': round(competitor_price, 2),
                'holiday': holiday,
                'current_sales': round(max(500, current_sales), 2)
            })
    df = pd.DataFrame(rows)
    df = df.sort_values(['store_id','month']).reset_index(drop=True)
    # create lag, rolling mean and pct change per store
    df['next_month_sales'] = df.groupby('store_id')['current_sales'].shift(-1)
    df['sales_lag_1'] = df.groupby('store_id')['current_sales'].shift(1)
    df['sales_ma_3'] = df.groupby('store_id')['current_sales'].rolling(window=3, min_periods=1).mean().reset_index(level=0, drop=True)
    df['sales_pct_change_1'] = df.groupby('store_id')['current_sales'].pct_change().fillna(0)
    df = df.dropna().reset_index(drop=True)
    # classification target
    df['increase_flag'] = (df['next_month_sales'] > df['current_sales']).astype(int)
    return df

df = generate_improved_sales_data(n_months=40, n_stores=10)

@st.cache_resource(show_spinner=False)
def train_improved_models(df):
    features = ['store_id','month_of_year','marketing_spend','price','competitor_price','holiday','current_sales','sales_lag_1','sales_ma_3','sales_pct_change_1']
    X = df[features].copy()
    y_reg = df['next_month_sales'].copy()
    y_clf = df['increase_flag'].copy()
    # preprocessing: one-hot store_id and month_of_year, scale numeric
    cat_cols = ['store_id','month_of_year']
    num_cols = ['marketing_spend','price','competitor_price','holiday','current_sales','sales_lag_1','sales_ma_3','sales_pct_change_1']
    preproc = ColumnTransformer([
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ])
    # Tuned RandomForest hyperparams for stronger performance
    reg_pipe = Pipeline([
        ('pre', preproc),
        ('reg', RandomForestRegressor(n_estimators=700, max_depth=12, random_state=42, n_jobs=-1))
    ])
    clf_pipe = Pipeline([
        ('pre', preproc),
        ('clf', RandomForestClassifier(n_estimators=700, max_depth=12, class_weight='balanced', random_state=42, n_jobs=-1))
    ])
    X_train, X_test, yreg_train, yreg_test, yclf_train, yclf_test = train_test_split(
        X, y_reg, y_clf, test_size=0.2, random_state=42, stratify=y_clf)
    reg_pipe.fit(X_train, yreg_train)
    clf_pipe.fit(X_train, yclf_train)
    # metrics
    yreg_pred = reg_pipe.predict(X_test)
    yclf_pred = clf_pipe.predict(X_test)
    reg_metrics = {'mae': mean_absolute_error(yreg_test, yreg_pred), 'r2': r2_score(yreg_test, yreg_pred)}
    clf_metrics = {'accuracy': accuracy_score(yclf_test, yclf_pred), 'report': classification_report(yclf_test, yclf_pred, output_dict=True), 'confusion_matrix': confusion_matrix(yclf_test, yclf_pred)}
    return {'reg_model': reg_pipe, 'clf_model': clf_pipe, 'features': features, 'reg_metrics': reg_metrics, 'clf_metrics': clf_metrics, 'classes': clf_pipe.classes_}

state = train_improved_models(df)

# Sidebar inputs
st.sidebar.header("Enter current month data (single record)")
store = st.sidebar.selectbox("Store", sorted(df['store_id'].unique()))
month_of_year = st.sidebar.selectbox("Month of Year", sorted(df['month_of_year'].unique()))
marketing_spend = st.sidebar.number_input("Marketing Spend", min_value=0.0, value=6000.0, step=100.0, format="%.2f")
price = st.sidebar.number_input("Price per unit", min_value=0.1, value=25.0, step=0.1, format="%.2f")
competitor_price = st.sidebar.number_input("Competitor Price", min_value=0.1, value=26.0, step=0.1, format="%.2f")
holiday = st.sidebar.selectbox("Holiday month (1=yes, 0=no)", [0,1])
current_sales = st.sidebar.number_input("Current Month Sales (amount)", min_value=0.0, value=50000.0, step=100.0, format="%.2f")

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
    # add engineered features from the recent history (approximate): use last known sales_lag_1 and ma_3 from df
    store_hist = df[df['store_id']==store].sort_values('month')
    last_row = store_hist.iloc[-1]
    rec['sales_lag_1'] = last_row['current_sales']
    rec['sales_ma_3'] = store_hist['current_sales'].rolling(window=3, min_periods=1).mean().iloc[-1]
    rec['sales_pct_change_1'] = (rec['current_sales'].values[0] - rec['sales_lag_1'].values[0]) / max(1, rec['sales_lag_1'].values[0])
    reg_model = state['reg_model']
    clf_model = state['clf_model']
    pred_sales = reg_model.predict(rec)[0]
    prob_increase = None
    if hasattr(clf_model, "predict_proba"):
        prob = clf_model.predict_proba(rec)[0]
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
        fig, ax = plt.subplots()
        ax.bar(prob_increase['class'].astype(str), prob_increase['probability'])
        ax.set_ylim(0,1)
        ax.set_ylabel("Probability")
        ax.set_title("Probability of classes (increase=1, decrease=0)")
        st.pyplot(fig)
    # sensitivity to marketing
    adj_low = rec.copy(); adj_low['marketing_spend'] = rec['marketing_spend']*0.9
    adj_high = rec.copy(); adj_high['marketing_spend'] = rec['marketing_spend']*1.1
    low_pred = reg_model.predict(adj_low)[0]
    high_pred = reg_model.predict(adj_high)[0]
    st.write("Quick sensitivity: effect of ±10% marketing spend")
    st.write(f"-10% marketing => predicted sales: {low_pred:,.2f}")
    st.write(f"+10% marketing => predicted sales: {high_pred:,.2f}")

# Show data and metrics
st.markdown("---")
st.subheader("Sample synthetic sales data (with engineered features)")
st.dataframe(df.sample(8, random_state=2).reset_index(drop=True))

st.subheader("Regression model metrics (test set)")
st.write(f"MAE: {state['reg_metrics']['mae']:.2f}")
st.write(f"R²: {state['reg_metrics']['r2']:.3f}")

st.subheader("Classification model metrics (test set)")
st.write(f"Accuracy: {state['clf_metrics']['accuracy']:.3f}")
report_df = pd.DataFrame(state['clf_metrics']['report']).transpose()
st.dataframe(report_df)
st.write("Confusion matrix (rows: true, columns: predicted)")
st.write(state['clf_metrics']['confusion_matrix'])

st.markdown("---")
st.write("Notes: This improved demo uses feature engineering (lags, rolling mean, pct change) and tuned Random Forests to create clearer signal in synthetic data. For production, use your real dataset and consider cross-validation, hyperparameter tuning, and model persistence.")
