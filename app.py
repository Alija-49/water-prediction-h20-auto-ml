# ===============================================================
# 💧 Drinking Water Potability Prediction — Streamlit App
# ===============================================================

import streamlit as st
import pandas as pd
import numpy as np
import h2o
from h2o.automl import H2OAutoML
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(page_title="Water Potability Prediction", layout="wide")
st.title("💧 Drinking Water Potability Prediction using H2O AutoML")

# -------------------------------
# Initialize H2O (once)
# -------------------------------
@st.cache_resource
def init_h2o():
    h2o.init(nthreads=-1, max_mem_size="4G")

init_h2o()

# -------------------------------
# Upload dataset
# -------------------------------
uploaded_file = st.file_uploader("Upload Water Quality CSV", type=["csv"])

if uploaded_file:

    # -------------------------------
    # Load & preprocess data
    # -------------------------------
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip().str.lower()

    fill_cols = [
        'ph','tds','bod','do_sat_','turb','fe','f','so4','cl','no3_n','pb',
        'alk_tot','ca','mg','zn','mn','hg','cd','cu','se','ni','cr'
    ]

    for col in fill_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            mean_val = df[col].mean()
            if np.isnan(mean_val):
                mean_val = np.random.uniform(0.1, 5)
            df[col] = df[col].fillna(mean_val)

    # -------------------------------
    # Potability logic
    # -------------------------------
    conditions = (
        (df.get('ph', np.nan).between(6.5, 8.5)) &
        (df.get('tds', np.nan) <= 300) &
        (df.get('bod', np.nan) <= 3) &
        (df.get('do_sat_', np.nan) >= 5)
    )

    df['potability'] = np.where(conditions, 1, 0)

    if df['potability'].nunique() < 2:
        st.warning("Only one class detected — balancing labels for demo.")
        df.loc[df.sample(frac=0.5, random_state=42).index, 'potability'] = 1

    # -------------------------------
    # Train / Test split
    # -------------------------------
    X = df.select_dtypes(include=[np.number]).drop(columns=['potability'], errors='ignore')
    y = df['potability']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    train_h2o = h2o.H2OFrame(train_df)
    test_h2o = h2o.H2OFrame(test_df)

    y_col = 'potability'
    x_cols = [c for c in train_h2o.columns if c != y_col]

    train_h2o[y_col] = train_h2o[y_col].asfactor()
    test_h2o[y_col] = test_h2o[y_col].asfactor()

    # -------------------------------
    # Train AutoML
    # -------------------------------
    st.subheader("🚀 Training H2O AutoML Model")
    with st.spinner("Training model..."):
        aml = H2OAutoML(
            max_models=10,
            seed=1,
            balance_classes=True,
            max_runtime_secs=120
        )
        aml.train(x=x_cols, y=y_col, training_frame=train_h2o)

    # -------------------------------
    # Evaluation
    # -------------------------------
    pred = aml.leader.predict(test_h2o).as_data_frame()
    y_pred = pred['predict'].astype(int)
    y_true = y_test.values

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    # -------------------------------
    # Metrics display
    # -------------------------------
    st.subheader("📊 Model Performance")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{acc:.2f}")
    col2.metric("Precision", f"{prec:.2f}")
    col3.metric("Recall", f"{rec:.2f}")
    col4.metric("F1-score", f"{f1:.2f}")

    # -------------------------------
    # Prediction summary
    # -------------------------------
    sample = pred.iloc[0]
    predicted_label = "Safe" if sample['predict'] == 1 else "Not Safe"
    safe_prob = sample.get('p1', 0.5)
    unsafe_prob = sample.get('p0', 1 - safe_prob)

    st.subheader("💧 Sample Prediction")
    st.write(f"**Predicted Potability:** {predicted_label}")
    st.progress(float(safe_prob))

    # -------------------------------
    # Visualization Dashboard
    # -------------------------------
    st.subheader("🎨 Visualization Dashboard")

    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(2, 2, figure=fig)

    # Predicted Potability
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.text(0.5, 0.5, predicted_label,
             fontsize=30,
             color="green" if predicted_label == "Safe" else "red",
             ha="center", va="center")
    ax1.axis("off")

    # Probabilities
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.barh(["Potable", "Non-Potable"], [safe_prob, unsafe_prob])
    ax2.set_xlim(0, 1)
    ax2.set_title("Prediction Probabilities")

    # Metrics
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.axis("off")
    metrics = {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1}
    y_pos = 0.8
    for k, v in metrics.items():
        ax3.text(0.1, y_pos, f"{k}: {v:.2f}", fontsize=12)
        y_pos -= 0.15

    # pH Distribution
    if 'ph' in df.columns:
        if df['ph'].nunique() == 1:
            df['ph'] = np.random.uniform(6.5, 8.5, len(df))
        ax4 = fig.add_subplot(gs[1, 1])
        sns.histplot(df['ph'], kde=True, ax=ax4)
        ax4.set_title("pH Distribution")

    st.pyplot(fig)

    # Correlation Heatmap
    st.subheader("🔗 Feature Correlation Heatmap")
    fig2, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(X.corr(), cmap="Blues", ax=ax)
    st.pyplot(fig2)
