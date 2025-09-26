import streamlit as st
from helpers.state_ops import ensure_image, ordered_keys, stem, set_current_by_index
from helpers.upload_download_functions import load_tif_masks_for_rec
from helpers.mask_editing_functions import append_masks_to_rec


def render_main():
    ss = st.session_state

    # ---- sensible defaults (avoid KeyErrors) ----
    ss.setdefault("images", {})
    ss.setdefault("current_key", None)
    ss.setdefault("image_uploader_nonce", 0)
    ss.setdefault("mask_uploader_nonce", 0)

    # ---------- Layout: 2 columns ----------
    col1, col2 = st.columns([2, 2])

    # ---------- LEFT: uploads & summary ----------
    with col1:
        # ---- single uploader: images & masks ----
        st.subheader("Upload images & masks here")
        up_key = f"u_all_np_{ss.get('uploader_nonce', 0)}"
        files = st.file_uploader(
            "Upload images (.png/.jpg/.tif) and masks with '_mask' suffix (.tif)",
            type=["png", "jpg", "jpeg", "tif", "tiff"],
            accept_multiple_files=True,
            key=up_key,
        )

        def _process_uploads(files):
            if not files:
                return
            # images first
            imgs = [f for f in files if not stem(f.name).endswith("_mask")]
            for f in imgs:
                ensure_image(f)
            ok = ordered_keys()
            if ok:
                set_current_by_index(len(ok) - 1)

            # then masks (require prior image; match by stem without '_mask')
            masks = [f for f in files if stem(f.name).endswith("_mask")]
            if masks and ss.images:
                stem_to_key = {stem(rec["name"]): k for k, rec in ss.images.items()}
                for f in masks:
                    base = stem(f.name)[:-5]  # drop "_mask"
                    k = stem_to_key.get(base)
                    if k is None:
                        continue
                    rec = ss.images[k]
                    m = load_tif_masks_for_rec(f, rec)
                    append_masks_to_rec(rec, m)
                    ss.current_key = k

        if files:
            _process_uploads(files)
            ss["uploader_nonce"] = ss.get("uploader_nonce", 0) + 1
            st.rerun()

        # ---- Summary table: image–mask pairs ----
        st.subheader("Uploaded image–mask pairs")
        ok = ordered_keys()
        if not ok:
            st.info("No images uploaded yet.")
        else:
            h1, h2, h3, h4, h5 = st.columns([4, 2, 2, 2, 2])
            h1.markdown("**Image**")
            h2.markdown("**Mask present**")
            h3.markdown("**Number of cells**")
            h4.markdown("**Labelled Masks**")
            h5.markdown("**Remove**")
            for k in ok:
                rec = ss.images[k]
                masks = rec.get("masks")
                n_labels = sum(l is not None for l in rec.get("labels", []))
                has_mask = bool(masks is not None and getattr(masks, "size", 0) > 0)
                n_cells = int(masks.shape[0]) if has_mask else 0
                c1, c2, c3, c4, c5 = st.columns([4, 2, 2, 2, 2])
                c1.write(rec["name"])
                c2.write("✅" if has_mask else "❌")
                c3.write(str(n_cells))
                c4.write(f"{n_labels}/{n_cells}")
                if c5.button("Remove", key=f"remove_{k}"):
                    del ss.images[k]
                    ok2 = ordered_keys()
                    ss.current_key = ok2[0] if ok2 else None
                    st.rerun()

    # ---------- RIGHT: model uploads (persist in session_state) ----------
    with col2:

        # Initialize model state once
        ss.setdefault("cellpose_model_bytes", None)
        ss.setdefault("cellpose_model_name", None)
        ss.setdefault("densenet_ckpt_bytes", None)
        ss.setdefault("densenet_ckpt_name", None)

        # ---- Cellpose model (custom weights) ----
        st.subheader("Upload Cellpose model here")
        cellpose_file = st.file_uploader(
            "Upload Cellpose model file",
            # Cellpose custom models are commonly .npy; allow pt/pth just in case
            type=["npy", "pt", "pth"],
            key="upload_cellpose_model",
        )
        if cellpose_file is not None:
            ss["cellpose_model_bytes"] = cellpose_file.read()
            ss["cellpose_model_name"] = cellpose_file.name
            st.success(f"Loaded Cellpose model: {cellpose_file.name}")

        # ---- DenseNet-121 classifier ----
        st.subheader("Upload Densenet121 classifier here")
        densenet_file = st.file_uploader(
            "Upload DenseNet-121 checkpoint",
            type=["pt", "pth", "ckpt", "bin", "keras"],
            key="upload_densenet_ckpt",
        )
        if densenet_file is not None:
            ss["densenet_ckpt_bytes"] = densenet_file.read()
            ss["densenet_ckpt_name"] = densenet_file.name
            st.success(f"Loaded DenseNet-121 checkpoint: {densenet_file.name}")

        # ---- Status panel ----
        st.subheader("**Current model files**")
        st.write("Cellpose model:", ss.get("cellpose_model_name") or "—")
        st.write("DenseNet-121 model:", ss.get("densenet_ckpt_name") or "—")

        # ---- Clear buttons ----
        col_a, col_b = st.columns(2)
        if col_a.button("Clear Cellpose model", use_container_width=True):
            ss["cellpose_model_bytes"] = None
            ss["cellpose_model_name"] = None
        if col_b.button("Clear DenseNet-121", use_container_width=True):
            ss["densenet_ckpt_bytes"] = None
            ss["densenet_ckpt_name"] = None
