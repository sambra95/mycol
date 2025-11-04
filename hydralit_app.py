# app_hydralit.py
import os
import runpy
import streamlit as st
from hydralit import HydraApp
import hydralit_components as hc

# ------------------ App setup ------------------ #
st.set_page_config(page_title="Mycoscope", page_icon="🧬", layout="wide")

# Your original boot steps
from boot import common_boot, configure_tf_cpu_only
from helpers.state_ops import ensure_global_state

ensure_global_state()
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
configure_tf_cpu_only()
common_boot()

# Optional: hide Streamlit's sidebar nav (since Hydralit provides top nav)
st.markdown(
    """
    <style>
      [data-testid="stSidebarNav"] { display: none; }
    </style>
    """,
    unsafe_allow_html=True,
)

import streamlit as st

st.markdown(
    """
<style>
/* Remove all outer padding */
div.block-container {
    padding-top: 0rem;
    padding-bottom: 0rem;
    padding-left: 1rem;
    padding-right: 1rem;
}

/* Optional: also remove header spacing if using st.navigation() */
section[data-testid="stHeader"] {
    padding: 0;
}

/* Optional: make the main container full-width */
main {
    padding: 0;
}
</style>
""",
    unsafe_allow_html=True,
)


# Helper to execute a standard Streamlit script as a "page"
def run_view(script_path: str):
    # Execute the target Streamlit script in its own namespace
    # Assumes those scripts do NOT call st.set_page_config again.
    runpy.run_path(script_path, run_name="__main__")


# ------------------ Hydralit app ------------------ #
app = HydraApp(
    title="Mycoscope",
    favicon="🧬",
    use_loader=False,
    hide_streamlit_markers=True,  # cleaner header
)


# Your four functional pages, mapped 1:1 to the original files
@app.addapp(title="Upload Models and Data", icon="📥")
def page_upload():
    run_view("views/1_Upload_data.py")


@app.addapp(title="Segment and Classify Cells", icon="🎭")
def page_segment_classify():
    run_view("views/2_Create_and_Edit_Masks.py")


@app.addapp(title="Train Segmentation and Classification Models", icon="🧠")
def page_train_models():
    run_view("views/4_Fine_Tune_Models.py")


@app.addapp(title="Analyze Cell Groups", icon="📊")
def page_metrics():
    run_view("views/5_Cell_Metrics.py")


# ------------------ Run ------------------ #

app.run()
