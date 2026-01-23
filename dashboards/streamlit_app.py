import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

@st.cache_data
def load_data():
    return pd.read_csv(
        os.path.join(BASE_DIR, "data", "processed", "churn_segmented.csv")
    )


# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="AI-Driven Customer Retention Dashboard",
    layout="wide"
)

st.title("📊 AI-Driven Customer Retention & Pricing Dashboard")
st.caption("Executive decision-support view")

# Load data & model

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

@st.cache_data
def load_clean_data():
    return pd.read_csv(
        os.path.join(BASE_DIR, "data", "processed", "churn_cleaned.csv")
    )

df_clean = load_clean_data()

@st.cache_resource
def load_model():
    return joblib.load(
        os.path.join(BASE_DIR, "models", "xgboost_churn_model.pkl")
    )
@st.cache_resource
def load_feature_names():
    return joblib.load(
        os.path.join(BASE_DIR, "models", "feature_names.pkl")
    )

feature_names = load_feature_names()


df = load_data()
model = load_model()

X_shap_full = df_clean[feature_names]
@st.cache_resource
def load_shap_explainer(_model):
    return shap.Explainer(_model)

explainer = load_shap_explainer(model)

# -----------------------------
# KPI ROW
# -----------------------------
st.subheader("📌 Key Business Metrics")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Customers", f"{df.shape[0]:,}")
col2.metric("Avg Monthly Revenue", f"${df['MonthlyCharges'].mean():.2f}")
col3.metric("Overall Churn Rate", f"{df['Churn'].mean()*100:.1f}%")
col4.metric("High-Risk Customers", f"{(df['churn_prob'] > 0.7).sum():,}")

st.divider()

# -----------------------------
# Churn Risk Distribution
# -----------------------------
st.subheader("📉 Customer Churn Risk Distribution")

st.bar_chart(df['churn_prob'].value_counts(bins=10))

st.divider()

# -----------------------------
# Segment-Level Insights
# -----------------------------
st.subheader("🧩 Segment-Level Performance")

segment_summary = df.groupby('segment').agg(
    churn_rate=('Churn', 'mean'),
    avg_monthly_revenue=('MonthlyCharges', 'mean'),
    avg_churn_probability=('churn_prob', 'mean'),
    customers=('segment', 'count')
).reset_index()

st.dataframe(segment_summary, use_container_width=True)

st.divider()

# -----------------------------
# Revenue Impact
# -----------------------------
st.subheader("💰 Revenue Impact of AI Strategy")

baseline_revenue = df['baseline_expected_revenue'].sum()
post_strategy_revenue = df['post_strategy_expected_revenue'].sum()
uplift = post_strategy_revenue - baseline_revenue
uplift_pct = uplift / baseline_revenue * 100

col1, col2 = st.columns(2)

col1.metric(
    "Baseline Expected Revenue",
    f"${baseline_revenue:,.2f}"
)

col2.metric(
    "Post-Strategy Revenue",
    f"${post_strategy_revenue:,.2f}",
    delta=f"{uplift_pct:.2f}%"
)

st.divider()

# -----------------------------
# High-Risk Customer Drilldown
# -----------------------------
st.subheader("🔍 High-Risk Customer Drilldown")

risk_threshold = st.slider(
    "Select churn risk threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.7
)

high_risk_df = df[df['churn_prob'] >= risk_threshold]

st.write(f"Customers above {risk_threshold:.2f} churn risk:")
st.dataframe(
    high_risk_df[
        ['segment', 'MonthlyCharges', 'tenure', 'churn_prob']
    ].sort_values('churn_prob', ascending=False).head(50),
    use_container_width=True
)

st.divider()

# -----------------------------
# Strategic Recommendations
# -----------------------------
st.subheader("📌 Executive Recommendations")

st.markdown("""
**Based on AI-driven insights:**

• 🎯 Offer targeted discounts to **price-sensitive, high-risk segments**  
• 🛠 Improve onboarding for **new customers with low tenure**  
• 💼 Upsell add-ons to **loyal, high-value customers**  
• 🚫 Avoid blanket discounts across the entire customer base  

**Estimated Impact:**  
✔ Reduced churn  
✔ Improved customer lifetime value  
✔ Positive revenue uplift
""")
st.divider()
st.subheader("🧠 Why Customers Churn (Explainable AI)")
X_shap = X_shap_full.sample(300, random_state=42)

shap_values = explainer(X_shap)
st.markdown("### 🔍 Global Churn Drivers")

fig, ax = plt.subplots()
shap.summary_plot(shap_values, X_shap, show=False)
st.pyplot(fig)
plt.clf()
st.markdown("### 📊 Feature Importance (Executive View)")

fig, ax = plt.subplots()
shap.summary_plot(shap_values, X_shap, plot_type="bar", show=False)
st.pyplot(fig)
plt.clf()
st.markdown("### 👤 Individual Customer Explanation")

customer_index = st.slider(
    "Select customer index",
    min_value=0,
    max_value=len(X_shap) - 1,
    value=0,
    key="shap_customer_slider"
)

fig, ax = plt.subplots()
customer_index = st.slider(
    "Select customer index",
    min_value=0,
    max_value=len(X_shap) - 1,
    value=0
)

shap.plots.waterfall(shap_values[customer_index], show=False)
st.pyplot(fig)
plt.clf()

