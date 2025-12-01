import io
import requests
import numpy as np
import pandas as pd
import streamlit as st
import pgeocode

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
        ### Dashboard built on a synthetic dataset
        #### For exploring potential prescription anomalies and risk patterns.
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

@st.cache_data(show_spinner=True)
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

    # Read Excel
    df = pd.read_excel(io.BytesIO(data_bytes), dtype=str, engine="openpyxl")

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
# Load data (with error display if needed)
# =========================================
try:
    with st.spinner("Loading data from private GitHub repo..."):
        df = load_data_from_github()
except Exception as e:
    st.error("❌ Failed to load the dataset.")
    st.exception(e)
    st.stop()

st.success("✅ Data loaded successfully.")

# =========================================
# Identify key columns
# =========================================
flag_cols = [c for c in df.columns if c.startswith("T/F")]
has_zip_distance_flag = "T/F Distance" in flag_cols
has_zip_distance_value = "Patient - Pharmacy ZIP Distance_num" in df.columns
has_impossible_days_flag = "T/F Impossible Days Supply" in flag_cols

total_records = len(df)
total_with_any_violation = int(df["AnyViolation"].sum()) if "AnyViolation" in df.columns else 0

# =========================================
# Sidebar – filters
# =========================================
st.sidebar.header("Filters")

# Violation flags selector
selected_flags = st.sidebar.multiselect(
    "Select violation flags to focus on",
    options=flag_cols,
    default=flag_cols,
    help="Rows are considered 'violations' if any of the selected flags is True."
)

# RiskScore threshold slider (optional)
if "RiskScore_num" in df.columns:
    max_risk = float(np.nanmax(df["RiskScore_num"]))
    risk_min = st.sidebar.slider(
        "Minimum RiskScore",
        min_value=0.0,
        max_value=max(10.0, max_risk),
        value=0.0,
        step=1.0,
    )
else:
    risk_min = 0.0

# Optional min ZIP distance filter (for global filtered view)
if has_zip_distance_value:
    max_zip = float(np.nanmax(df["Patient - Pharmacy ZIP Distance_num"]))
    zip_min_filter = st.sidebar.slider(
        "Minimum Patient–Pharmacy ZIP Distance (miles)",
        min_value=0.0,
        max_value=max(10.0, max_zip),
        value=0.0,
        step=5.0,
    )
else:
    zip_min_filter = 0.0

# Apply filters to create filtered_df (for bottom preview)
filtered_df = df.copy()
if selected_flags:
    filtered_df = filtered_df[filtered_df[selected_flags].any(axis=1)]
if "RiskScore_num" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["RiskScore_num"] >= risk_min]
if has_zip_distance_value and zip_min_filter > 0:
    filtered_df = filtered_df[filtered_df["Patient - Pharmacy ZIP Distance_num"] >= zip_min_filter]

st.sidebar.write(f"Rows after filters: **{len(filtered_df):,}**")

# =========================================
# Overview metrics
# =========================================
st.subheader("Overview")

col_m1, col_m2, col_m3, col_m4 = st.columns(4)

unique_patients = df["Patient Last Name"].nunique() if "Patient Last Name" in df.columns else np.nan
unique_prescribers = df["Prescriber DEA"].nunique() if "Prescriber DEA" in df.columns else np.nan

col_m1.metric("Total Records", f"{total_records:,}")
col_m2.metric("Records with Any Violation", f"{total_with_any_violation:,}")
col_m3.metric("Unique Patients", f"{unique_patients:,}" if pd.notna(unique_patients) else "-")
col_m4.metric("Unique Prescribers", f"{unique_prescribers:,}" if pd.notna(unique_prescribers) else "-")

# =========================================
# Violation summary table
# =========================================
st.subheader("Violation Summary by Flag")

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

# =========================================
# ZIP Distance violations (histogram + map)
# =========================================
st.subheader("ZIP Distance Violations")

if has_zip_distance_flag and has_zip_distance_value:
    viol_zip_df = df[df["T/F Distance"]].copy()
    st.markdown(
        f"- Records flagged as **distance-related violations** (`T/F Distance = True`): "
        f"**{len(viol_zip_df):,}**"
    )

    # Histogram of distances
    dist_col = "Patient - Pharmacy ZIP Distance_num"
    dist_series = viol_zip_df[dist_col].dropna().astype(float)

    if not dist_series.empty:
        st.markdown("**Distribution of Patient–Pharmacy ZIP Distances (miles)**")
        # Use bins for reasonable bar_chart
        hist = dist_series.value_counts(bins=20).sort_index()
        hist.index = hist.index.astype(str)
        st.bar_chart(hist)
    else:
        st.info("No valid numeric ZIP distances found for histogram.")

    # Map of patient ZIP locations
    st.markdown("**Map of Patient ZIPs in Distance Violations (approximate locations)**")

    if "Patient Zip" in viol_zip_df.columns:
        nomi = pgeocode.Nominatim("us")

        @st.cache_data(show_spinner=False)
        def zip_to_latlon(zips):
            zips = pd.Series(zips).dropna().astype(str).str[:5]
            zips_unique = zips.unique()
            locations = nomi.query_postal_code(list(zips_unique))
            loc_df = pd.DataFrame({
                "zip": zips_unique,
                "lat": locations["latitude"].values,
                "lon": locations["longitude"].values,
            })
            loc_df = loc_df.dropna(subset=["lat", "lon"])
            return loc_df

        patient_zip_df = zip_to_latlon(viol_zip_df["Patient Zip"])
        if not patient_zip_df.empty:
            st.map(patient_zip_df.rename(columns={"lat": "latitude", "lon": "longitude"}))
        else:
            st.info("Could not resolve any patient ZIP codes to coordinates for mapping.")
    else:
        st.info("Column 'Patient Zip' not found; cannot generate map.")

else:
    st.info("ZIP distance violation data (`T/F Distance` and/or `Patient - Pharmacy ZIP Distance`) not available.")

# =========================================
# Impossible Days Supply – explanation & examples
# =========================================
st.subheader("Impossible Days Supply – Explanation & Examples")

if has_impossible_days_flag:
    imp_df = df[df["T/F Impossible Days Supply"]].copy()
    count_imp = len(imp_df)

    st.markdown(
        f"- Records flagged as **Impossible Days Supply** "
        f"(`T/F Impossible Days Supply = True`): **{count_imp:,}**"
    )

    with st.expander("How 'Impossible Days Supply' is computed (conceptual rules)"):
        st.markdown(
            """
            The **'T/F Impossible Days Supply'** flag identifies prescriptions where the
            reported **Days’ Supply** is unlikely or inconsistent, given the quantity and
            general real-world constraints. Typical checks include:
            
            - **Missing or zero Days’ Supply with positive Quantity**  
              - Example: 30 tablets dispensed but Days’ Supply is 0 or blank.
            - **Days’ Supply greater than a maximum practical limit**  
              - For example, more than **365 days** for any drug.  
              - Or more than **90 days** for certain controlled substances
                (e.g., opioids, benzodiazepines, stimulants).
            - **Implausible Quantity-per-Day**  
              - Compute `qty_per_day = Quantity Dispensed / Days’ Supply`.  
              - Flag if this is **very high** (e.g., > 10 units per day) or **very low**
                (e.g., < 0.1), suggesting the Days’ Supply was mis-entered.
            - **Inconsistent combinations**  
              - Very short Days’ Supply with many refills authorized.  
              - Very long Days’ Supply for medications usually given in smaller intervals.
            
            These rules simulate data quality and safety checks used by pharmacies,
            payers, and regulatory bodies when reviewing prescription claims.
            """
        )

    # Sample rows
    max_rows_imp = st.slider(
        "Number of flagged 'Impossible Days Supply' rows to show",
        min_value=5,
        max_value=200,
        value=20,
        step=5,
    )

    cols_to_show = [
        "Rx Number",
        "Date Written",
        "Date Filled",
        "Quantity Dispensed",
        "Days’ Supply",
        "Drug Name",
        "DEA Category",
        "RiskScore",
        "T/F Impossible Days Supply",
        "Anomaly",
    ]
    cols_to_show = [c for c in cols_to_show if c in imp_df.columns]
    st.dataframe(imp_df[cols_to_show].head(max_rows_imp), use_container_width=True)

else:
    st.info("No 'T/F Impossible Days Supply' flag found in the dataset.")

# =========================================
# Filtered data preview (based on sidebar selections)
# =========================================
st.subheader("Filtered Records (Based on Selected Flags / RiskScore / ZIP Distance)")

max_rows_preview = st.slider(
    "Max rows to display in preview",
    min_value=50,
    max_value=2000,
    value=200,
    step=50,
)

st.dataframe(filtered_df.head(max_rows_preview), use_container_width=True)
