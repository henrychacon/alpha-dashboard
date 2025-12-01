import streamlit as st
import requests
import io
import pandas as pd

def load_private_excel():
    token  = st.secrets["GITHUB_TOKEN"]
    user   = st.secrets["GITHUB_USER"]
    repo   = st.secrets["GITHUB_REPO"]
    branch = st.secrets["GITHUB_BRANCH"]
    path   = st.secrets["GITHUB_DATA_PATH"]

    url = f"https://raw.githubusercontent.com/{user}/{repo}/{branch}/{path}"

    headers = {"Authorization": f"token {token}"}
    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        st.error(f"Error loading data: {response.status_code}")
        return None

    return pd.read_excel(io.BytesIO(response.content))

# Header
header_col1, header_col2 = st.columns([1,4])

with header_col1:
    st.image("logo_alpha_data.png", use_column_width=True)

with header_col2:
    st.markdown("# Alpha Data\n**Insight That Protects.**")

st.write("Your dashboard is set up successfully. Add full code later.")
