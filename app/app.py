import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

st.set_page_config(page_title="Sales Touch Impact Dashboard", layout="wide")

st.title("Sales Touch Impact on YTD Orders")
st.write("Analyzing the effect of sales outreach on merchant performance")

@st.cache_data
def load_data():
    root = Path().resolve().parents[0] if Path().resolve().name == "notebooks" else Path().resolve()
    dfr = root / "data" / "processed" / "analysis_ready.csv"
    df = pd.read_csv(dfr, low_memory=False)
    return df

df = load_data()

st.write("### Key Results")

col1, col2, col3 = st.columns(3)

raw_gap = 8246.48
att = 1059.45
bias_reduction = ((raw_gap - att) / raw_gap) * 100

col1.metric(label="Raw Gap (Before Matching)", value=f"{raw_gap:,.0f} orders")
col2.metric(label="ATT (After Matching)", value=f"{att:,.0f} orders")
col3.metric(label="Bias Removed", value=f"{bias_reduction:.0f}%")

st.write("---")
st.write("### Effect by Account Size")

quantile_data = pd.DataFrame({
    "Quantile": ["10th", "50th", "90th"],
    "Coefficient": [0.00, 111.67, 263.22]
})

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(quantile_data["Quantile"], quantile_data["Coefficient"], color=["blue", "green", "red"])
ax.set_ylabel("Coefficient for HAS_SALES_TOUCH")
ax.set_title("Quantile Regression Coefficients")
st.pyplot(fig)

st.write("""
- **10th percentile** – no effect
- **50th percentile** – 112 orders
- **90th percentile** – 263 orders
""")

st.write("---")
st.write("### Effect by Order Volume Cluster")

cluster_data = pd.DataFrame({
    "Cluster": ["High Volume", "Low Volume"],
    "ATT": [264845, 717]
})

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(cluster_data["Cluster"], cluster_data["ATT"], color=["darkred", "steelblue"])
ax.set_ylabel("ATT (Orders)")
ax.set_title("ATT by Order Volume Cluster")
st.pyplot(fig)

st.write("""
- **High volume accounts** – 264,845 additional YTD orders
- **Low volume accounts** – 717 additional YTD orders
""")

st.write("---")
st.write("### Raw vs Matched Comparison")

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(["Full Data (Unmatched)", "Matched Data"], [482.89, 227.42])
ax.set_ylabel("Coefficient for HAS_SALES_TOUCH")
ax.set_title("Effect of Sales Touch: Full vs Matched")
st.pyplot(fig)

st.write("---")
st.write("### Conclusion")
st.write("""
Sales touches are associated with an additional **1,059 YTD orders** after controlling for selection bias.
The raw gap of **8,246 orders** is misleading, as reps target the busiest accounts.
The effect is strongest for **high-volume accounts** – they get 264,845 additional orders vs only 717 for low-volume accounts.
**Recommendation:** Focus sales outreach on medium-to-large accounts where the lift is highest.
""")

st.write("---")
st.write("Data source: Veeqo Salesforce data (84,031 accounts, 124 columns)")