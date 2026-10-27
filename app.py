
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="FraudGuard AI",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ FraudGuard AI")
st.subheader("Credit Card Fraud Detection & Model Intelligence")
st.caption("Machine Learning Dashboard | Transaction Risk Analytics")

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "creditcard.csv"
RESULT_FILE = BASE_DIR / "model_comparison.csv"

# Load available project files
@st.cache_data
def load_data(path):
    return pd.read_csv(path)

@st.cache_data
def load_results(path):
    return pd.read_csv(path)

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Choose a page",
    ["Dashboard", "Transaction Explorer", "Model Comparison", "About"]
)

# Load dataset if available
df = None
if DATA_FILE.exists():
    try:
        df = load_data(DATA_FILE)
    except Exception as e:
        st.error(f"Could not read creditcard.csv: {e}")

# Dashboard
if page == "Dashboard":
    st.header("Project Overview")

    if df is not None and "Class" in df.columns:
        total = len(df)
        frauds = int(df["Class"].sum())
        legitimate = total - frauds
        fraud_rate = frauds / total * 100 if total else 0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Transactions", f"{total:,}")
        c2.metric("Fraudulent Transactions", f"{frauds:,}")
        c3.metric("Legitimate Transactions", f"{legitimate:,}")
        c4.metric("Fraud Rate", f"{fraud_rate:.3f}%")

        left, right = st.columns(2)

        with left:
            st.subheader("Transaction Class Distribution")
            chart_data = pd.DataFrame({
                "Transaction Type": ["Legitimate", "Fraud"],
                "Count": [legitimate, frauds]
            })
            fig = px.bar(
                chart_data,
                x="Transaction Type",
                y="Count",
                color="Transaction Type",
                text="Count",
                title="Legitimate vs Fraudulent Transactions"
            )
            st.plotly_chart(fig, use_container_width=True)

        with right:
            st.subheader("Transaction Amount Distribution")
            fig = px.histogram(
                df,
                x="Amount",
                color="Class",
                nbins=50,
                title="Transaction Amounts by Class",
                labels={"Class": "Class (0 = Legitimate, 1 = Fraud)"}
            )
            st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("Place creditcard.csv in the same folder as app.py to display dashboard statistics.")

    st.info(
        "This dashboard explores the dataset and compares trained ML models. "
        "Dataset labels are historical labels, not live bank transaction decisions."
    )

# Transaction Explorer
elif page == "Transaction Explorer":
    st.header("Transaction Explorer")
    st.write("Inspect transactions from your dataset or upload a CSV file.")

    uploaded_file = st.file_uploader("Upload transaction CSV", type=["csv"])

    data = None
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Could not read uploaded CSV: {e}")
    elif df is not None:
        data = df

    if data is not None:
        st.write(f"**Rows:** {len(data):,} | **Columns:** {len(data.columns)}")
        st.dataframe(data.head(100), use_container_width=True)

        if "Class" in data.columns:
            st.subheader("Class Summary")
            summary = data["Class"].value_counts().rename(
                index={0: "Legitimate", 1: "Fraud"}
            )
            st.bar_chart(summary)

        st.download_button(
            "Download displayed dataset",
            data=data.to_csv(index=False).encode("utf-8"),
            file_name="transactions_export.csv",
            mime="text/csv"
        )
    else:
        st.warning("No dataset found. Upload a CSV or add creditcard.csv to your project folder.")

# Model Comparison
elif page == "Model Comparison":
    st.header("Machine Learning Model Comparison")

    if RESULT_FILE.exists():
        try:
            results = load_results(RESULT_FILE)
            st.dataframe(results, use_container_width=True)

            numeric_cols = results.select_dtypes(include="number").columns.tolist()
            metric_candidates = [
                "F1 Score", "F1", "PR-AUC", "Average Precision",
                "Recall", "Precision", "ROC-AUC", "Accuracy"
            ]
            metric = next(
                (col for col in metric_candidates if col in results.columns),
                numeric_cols[0] if numeric_cols else None
            )

            if metric and len(results.columns) > 0:
                model_col = next(
                    (col for col in results.columns
                     if col.lower() in ["model", "models", "model name", "algorithm"]),
                    None
                )

                if model_col:
                    chart = results.sort_values(metric, ascending=False)
                    fig = px.bar(
                        chart,
                        x=model_col,
                        y=metric,
                        title=f"Model Comparison — {metric}",
                        text_auto=".3f"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.caption(
                        "The results file does not have a recognisable model-name column. "
                        "Review the table above."
                    )
        except Exception as e:
            st.error(f"Could not read model_comparison.csv: {e}")
    else:
        st.warning(
            "model_comparison.csv was not found. Run compare_models.py first "
            "to generate the model comparison results."
        )

    st.caption(
        "For fraud detection, compare precision, recall, F1-score and PR-AUC. "
        "Accuracy alone can be misleading because fraud cases are rare."
    )

# About
elif page == "About":
    st.header("About FraudGuard AI")
    st.markdown("""
    **Objective:** Explore credit-card transactions and compare machine-learning
    approaches for identifying potentially fraudulent transactions.

    **Project components**
    - Dataset exploration and class-imbalance analysis
    - Transaction filtering and CSV export
    - Trained model comparison
    - Visual analytics for fraud detection metrics

    **Important:** This interface does not make live banking decisions or
    connect to a payment processor. Historical dataset labels are not predictions.
    """)

st.sidebar.divider()
st.sidebar.caption("FraudGuard AI | Academic ML Project")
