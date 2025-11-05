import streamlit as st
from boot import common_boot
from panels import cell_metrics_panel

common_boot()

with st.container(border=True):
    cell_metrics_panel.render_sidebar()

st.divider()

cell_metrics_panel.render_main()
