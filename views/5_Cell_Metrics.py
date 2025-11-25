import streamlit as st
from panels import cell_metrics_panel

# Warning if no images have been uploaded yet
if st.session_state["images"] == {}:
    st.warning("⚠️ Please upload an image on the 'Upload Models and Data' tab first.")
    st.stop()

with st.container(border=True):
    cell_metrics_panel.render_plotting_options()

st.divider()

if st.button("Generate Plots"):
    cell_metrics_panel.render_plotting_main()
