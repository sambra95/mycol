# panels/downloads.py
import streamlit as st

from helpers.state_ops import ordered_keys
from helpers.densenet_functions import load_labeled_patches_from_session
from helpers.upload_download_functions import (
    build_model_artifacts_zip,
    build_densenet_artifacts_zip,
    build_masks_images_zip,
    build_patchset_zip_from_session,
    _build_cell_metrics_zip,
)


def render_main():
    st.markdown("## Downloads")

    images = st.session_state.get("images", {})
    ok = ordered_keys() if images else []

    with st.expander("Download models and performance metrics", expanded=False):

        def _cellpose_zip(fingerprint: tuple) -> bytes:
            return build_model_artifacts_zip("cellpose") or b""

        def _densenet_zip(fingerprint: tuple) -> bytes:
            return build_densenet_artifacts_zip() or b""

        # lightweight fingerprints so cache invalidates when inputs change
        cp_df = st.session_state.get("cp_grid_results_df")
        cp_fp = (
            len(st.session_state.get("cellpose_model_bytes") or b""),
            len(st.session_state.get("cp_losses_png") or b""),
            len(st.session_state.get("cp_compare_iou_png") or b""),
            tuple(getattr(cp_df, "shape", (0, 0))),
        )
        dn_fp = (
            len(st.session_state.get("densenet_model_bytes") or b""),
            len(st.session_state.get("densenet_plot_losses_png") or b""),
            len(st.session_state.get("densenet_plot_confusion_png") or b""),
        )

        c1, c2 = st.columns(2)

        with c1:
            st.download_button(
                "Download Cellpose artifacts (.zip)",
                data=_cellpose_zip(cp_fp),
                file_name="cellpose_artifacts.zip",
                mime="application/zip",
                use_container_width=True,
                key="dl_cellpose_zip",
            )

        with c2:
            st.download_button(
                "Download DenseNet artifacts (.zip)",
                data=_densenet_zip(dn_fp),
                file_name="densenet_artifacts.zip",
                mime="application/zip",
                use_container_width=True,
                key="dl_densenet_zip",
            )

    # Masks & images (with overlay options)
    with st.expander(
        "Download masks, labelled images, cell patches and labels", expanded=True
    ):
        c1, c2 = st.columns(2)
        include_overlay = c1.checkbox(
            "Include colored mask overlays", value=True, key="dl_include_overlay"
        )
        include_counts = c2.checkbox(
            "Overlay per-image class counts", value=False, key="dl_include_counts"
        )

        if not ok:
            st.info("No images in memory. Upload data first.")
        else:
            if st.button(
                "Build Masks & Images ZIP",
                use_container_width=True,
                key="build_masks_images_zip",
            ):
                st.session_state["masks_images_zip_ready"] = build_masks_images_zip(
                    images, ok, include_overlay, include_counts
                )

            data = st.session_state.get("masks_images_zip_ready")
            st.download_button(
                "Download masks & images (.zip)",
                data=data or b"",
                file_name="masks_and_images.zip",
                mime="application/zip",
                disabled=not bool(data),
                use_container_width=True,
                key="dl_masks_images_zip",
            )
        ps = int(st.session_state.get("densenet_patch_size", 64))
        X_tmp, _, _ = load_labeled_patches_from_session(patch_size=ps)
        if X_tmp.shape[0] == 0:
            st.info("No labeled patches available from the current session.")
            disabled = True
        else:
            disabled = False
            if st.button(
                "Build Patch Set ZIP", use_container_width=True, key="build_patch_zip"
            ):
                st.session_state["patch_zip_ready"] = build_patchset_zip_from_session(
                    patch_size=ps
                )

        data = st.session_state.get("patch_zip_ready")
        st.download_button(
            "Download patch set (.zip)",
            data=data or b"",
            file_name="cell_patches.zip",
            mime="application/zip",
            disabled=disabled or not bool(data),
            use_container_width=True,
            key="dl_patch_zip",
        )

    # Morphology plots (expects files or (name,bytes) in session_state['morphology_plots'])
    with st.expander(
        "Download class morphology plots and image cells counts", expanded=False
    ):

        # --- single button ---
        st.download_button(
            "Download cell metrics (.zip)",
            data=_build_cell_metrics_zip(
                tuple(st.session_state.get("analysis_labels") or ())
            ),
            file_name="cell_metrics.zip",
            mime="application/zip",
            use_container_width=True,
            key="dl_cell_metrics_zip",
        )
