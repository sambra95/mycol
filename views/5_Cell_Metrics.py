import streamlit as st
from boot import common_boot
from panels import cell_metrics_panel

common_boot()

with st.container(border=True):
    cell_metrics_panel.render_sidebar()  # if you have one

st.divider()

cell_metrics_panel.render_main()
