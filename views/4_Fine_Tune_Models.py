import streamlit as st
from boot import common_boot
from panels import fine_tune_panel

common_boot()

st.title("🧠 Fine Tune Segmentation and Classification Models")
st.divider()

col1, col2 = st.columns([1, 1])

# Main content (your two sections)
with col1:
    with st.container(border=True, height=450):
        fine_tune_panel.render_cellpose_train_panel()
with col2:
    with st.container(border=True, height=450):
        fine_tune_panel.render_densenet_train_panel()


def show_cellpose_training_plots(height: int = 600):
    """Render saved Cellpose plots from session state (if available)."""
    k1, k2 = "cp_losses_png", "cp_compare_iou_png"
    with st.container(border=True, height=height):
        if (k1 not in st.session_state) and (k2 not in st.session_state):
            st.empty()
            return
        st.header("Cellpose Training plots")
        if k1 in st.session_state:
            st.image(
                st.session_state[k1],
                use_container_width=True,
            )
        else:
            st.info("No fine-tuning data to show.")
        if k2 in st.session_state:
            st.image(
                st.session_state[k2],
                use_container_width=True,
            )
        else:
            st.info("No fine-tuning data to show.")


def show_densenet_training_plots(height: int = 600):
    """Render saved DenseNet training plots from session state (if available)."""
    k1, k2 = "densenet_plot_losses_png", "densenet_plot_confusion_png"
    with st.container(border=True, height=height):
        if (k1 not in st.session_state) and (k2 not in st.session_state):
            st.empty()
            return
        st.header("DenseNet Training Plots")
        if k1 in st.session_state:
            st.image(
                st.session_state[k1],
                use_container_width=True,
            )
        else:
            st.empty()
        if k2 in st.session_state:
            st.image(
                st.session_state[k2],
                use_container_width=True,
            )
        else:
            st.empty()


st.divider()
st.header("Most recent training results:")
col1, col2 = st.columns([1, 1])
with col1:
    show_cellpose_training_plots(height=700)
with col2:
    show_densenet_training_plots(height=700)
