import streamlit as st

# Header
header_col1, header_col2 = st.columns([1,4])

with header_col1:
    st.image("logo_alpha_data.png", use_column_width=True)

with header_col2:
    st.markdown("# Alpha Data\n**Insight That Protects.**")

st.write("Your dashboard is set up successfully. Add full code later.")
