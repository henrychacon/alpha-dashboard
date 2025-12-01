import io
import requests
import numpy as np
import pandas as pd
import streamlit as st

# =========================================
# Page config
# =========================================
st.set_page_config(
    page_title="Alpha Data – Insight That Protects.",
    layout="wide"
)

# =========================================
# Header with logo + title
# =========================================
header_col1, header_col2 = st.columns([1, 4])

with header_col1:
    # Logo file must be in the same repo as app.py
    try:
        st.image("logo_alpha_data.png", use_column_width=True)
    except Exception as e:
        st.warning(f"Logo not found or failed to load: {e}")

with header_col2:
    st.markdown(
        """
        # Alpha Data  
        **Insight That Protects.**
        """,
        unsafe_allow_html=True,
    )

st.caption(
    "Dashboard built on the synthetic dataset `synthetic_data_Final_Flags_and_Risk.xlsx` "
    "for exploring potential prescription anomalies and risk patterns."
)

# =========================================
# Data loader (from private GitHub repo via secrets)
# =========================================

def load_data_from_github():
    """
    Load Excel file from a private GitHub repository using a personal access token.

    Requires in st.secrets:
      - GITHUB_TOKEN
      - GITHUB_USER
      - GITHUB_REPO
      - GITHUB_BRANCH
      - GITHUB_DATA_PATH
    """
    # Basic sanity check of secrets
    required_keys = [
        "GITHUB_TOKEN",
        "GITHUB_USER",
        "GITHUB_REPO",
        "GITHUB_BRANCH",
        "GITHUB_DATA_PATH",
    ]
    missing = [k for k in required_keys if k not in st.secrets]
    if missing:
        raise RuntimeError(f"Missing keys in st.secrets: {missing}")

    token  = st.secrets["GITHUB_TOKEN"]
    user   = st.secrets["GITHUB_USER"]
    repo   = st.secrets["GITHUB_REPO"]
    branch = st.secrets.get("GITHUB_BRANCH", "main")
    path   = st.secrets["GITHUB_DATA_PATH"]

    url = f"https://raw.githubusercontent.com/{user}/{repo}/{branch}/{path}"

    headers = {"Authorization": f"token {token}"}
    resp = requests.get(url, headers=headers)

    if resp.status_code != 200:
        raise RuntimeError(
            f"Failed to load data from GitHub (status {resp.status_code}). "
            f"URL: {url} – Response snippet: {resp.text[:200]}"
        )

    data_bytes = resp.content

    # IMPORTANT: Excel read can fail if openpyxl not installed
    try:
        df = pd.read_excel(io.BytesIO(data_bytes), dtype=str, engine="openpyxl")
    except Exception as e:
        raise RuntimeError(f"Error reading Excel with pandas/openpyxl: {e}")

    # Basic numeric conversions
    num_cols = [
        "RiskScore",
        "Quantity Dispensed",
        "Days’ Supply",
        "Patient - Pharmacy ZIP Distance",
    ]
    for col in num_cols:
        if col in df.columns:
            df[col + "_num"] = pd.to_numeric(df[col], errors="coerce")

    # Convert T/F columns to proper booleans
    flag_cols = [c for c in df.columns if c.startswith("T/F")]
    for c in flag_cols:
        df[c] = df[c].astype(str).str.strip().str.lower().isin(
            ["true", "1", "yes", "y"]
        )

    # AnyViolation column
    if flag_cols:
        df["AnyViolation"] = df[flag_cols].any(axis=1)
    else:
        df["AnyViolation"] = False

    return df


# =========================================
# Try to load data and show clear errors
# =========================================
try:
    with st.spinner("Loading data from private GitHub repo..."):
        df = load_data_from_github()
except Exception as e:
    st.error("❌ Failed to load the dataset.")
    st.exception(e)  # show full traceback in the UI
    st.stop()

st.success("✅ Data loaded successfully.")

st.write("**Preview of columns and first 5 rows (for debugging):**")
st.write(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
st.dataframe(df.head())

# =========================================
# From here down is the actual dashboard logic
# (kept simpler at first to make sure loading works)
# =========================================

flag_cols = [c for c in df.columns if c.startswith("T/F")]
total_records = len(df)
total_with_any_violation = int(df["AnyViolation"].sum()) if "AnyViolation" in df.columns else 0

st.subheader("Overview")
c1, c2 = st.columns(2)
c1.metric("Total Records", f"{total_records:,}")
c2.metric("Records with Any Violation", f"{total_with_any_violation:,}")

st.subheader("Violation Flags Summary")
if flag_cols:
    summary_rows = []
    for col in flag_cols:
        count = int(df[col].sum())
        summary_rows.append({
            "Violation Flag": col,
            "Count (True)": count,
            "Percent of Records": (count / total_records * 100.0) if total_records > 0 else 0.0,
        })

    viol_summary = (
        pd.DataFrame(summary_rows)
        .sort_values("Count (True)", ascending=False)
        .reset_index(drop=True)
    )
    viol_summary["Percent of Records"] = viol_summary["Percent of Records"].round(2)
    st.dataframe(viol_summary, use_container_width=True)
else:
    st.info("No T/F violation flag columns were found in the dataset.")

st.subheader("Filtered Records (simple preview)")

selected_flags = st.multiselect(
    "Filter to records where these violation flags are True (any of them):",
    options=flag_cols,
    default=flag_cols,
)

filtered_df = df.copy()
if selected_flags:
    filtered_df = filtered_df[filtered_df[selected_flags].any(axis=1)]

max_rows_preview = st.slider(
    "Max rows to display in preview",
    min_value=50,
    max_value=2000,
    value=200,
    step=50,
)
st.dataframe(filtered_df.head(max_rows_preview), use_container_width=True)
