import os
import hashlib
import tempfile
import numpy as np
import streamlit as st
import cv2
from cellpose import core, models
from PIL import Image
from src.helpers.state_ops import (
    ordered_keys,
    get_current_rec,
    normalize_image,
)

# -----------------------------------------------------#
# ---------------- IMAGE PREPROCESSING --------------- #
# -----------------------------------------------------#


def preprocess_for_cellpose(rec):
    """takes record input and prepares the stored image for cellpose"""

    img = rec["image"]

    # convert to grayscale if needed
    if img.ndim == 3:
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    elif img.ndim != 2:
        raise ValueError(
            f"Unsupported image shape {img.shape}; expected (H,W) or (H,W,C)"
        )

    # normalize
    im_in = normalize_image(img)

    return im_in


def convert_cellpose_mask_to_single_array(mask_output, H, W):
    """Converts Cellpose output mask to single (H,W) label image with contiguous ids 1..N"""

    # handle empty mask case
    if mask_output is None or mask_output.size == 0:
        inst = np.zeros((H, W), dtype=np.uint8)
        return inst
    # handle standard case
    else:
        a = np.asarray(mask_output)
        if a.shape != (H, W):
            # (rare) ensure correct size; nearest preserves labels
            a = np.array(
                Image.fromarray(a).resize((W, H), Image.NEAREST), dtype=a.dtype
            )
        # remap ids to contiguous 1..K
        vals = np.unique(a)
        ids = vals[vals > 0]
        if ids.size == 0:
            inst = np.zeros((H, W), dtype=np.uint8)
        else:
            # remap old ids -> 1..K (contiguous)
            K = int(ids.size)
            max_old = int(a.max())
            lut_dtype = np.uint32 if K > np.iinfo(np.uint16).max else np.uint16
            lut = np.zeros(max_old + 1, dtype=lut_dtype)
            lut[ids] = np.arange(1, K + 1, dtype=lut_dtype)
            inst = lut[a]

        return inst


# -----------------------------------------------------#
# ---------------- CELLPOSE INFERENCE ---------------- #
# -----------------------------------------------------#


def get_cellpose_weights() -> str | None:
    """writes Cellpose model bytes from session state to a temp file and returns the path"""
    ss = st.session_state
    b = ss.get("cellpose_model_bytes", None)
    name = ss.get("cellpose_model_name", None)
    if not b or not name:
        return None

    h = hashlib.sha1(b).hexdigest()[:12]
    suffix = os.path.splitext(name)[1] or ".npy"
    path = os.path.join(tempfile.gettempdir(), f"cellpose_{h}{suffix}")

    # write once; if the file exists, assume it matches the hash
    if not os.path.exists(path):
        with open(path, "wb") as f:
            f.write(b)
    return path


def get_cellpose_model():
    ss = st.session_state
    tag = (
        hashlib.sha1(ss["cellpose_model_bytes"]).hexdigest()[:12]
        if ss.get("cellpose_model_bytes")
        else "cyto2"
    )

    if ss.get("cellpose_model_obj") is not None and ss.get("cellpose_model_tag") == tag:
        return ss["cellpose_model_obj"]

    weights_path = get_cellpose_weights()
    model_type = "cyto2"
    if weights_path:
        model_type = weights_path

    model = models.CellposeModel(
        gpu=core.use_gpu(),
        pretrained_model=model_type,
    )

    ss["cellpose_model_obj"] = model
    ss["cellpose_model_tag"] = tag

    return model


def segment_with_cellpose(
    rec: dict,
    *,
    channels=(0, 0),
    diameter=None,
    cellprob_threshold=-0.2,
    flow_threshold=0.4,
    min_size=0,
    niter=0,
) -> dict:
    """
    Runs Cellpose on rec['image'] and overwrites rec['masks'] with a single (H,W)
    integer label image (0=background, 1..N=instances). Resets rec['labels'].
    """

    im_in = preprocess_for_cellpose(rec)

    cell_model = get_cellpose_model()

    # reset diameter to None for automatic estimation
    if diameter == 0:
        diameter = None

    masks_out, flows, styles = cell_model.eval(
        [im_in],
        channels=list(channels),
        diameter=diameter,
        cellprob_threshold=cellprob_threshold,
        flow_threshold=flow_threshold,
        min_size=min_size,
        niter=niter,
    )
    mask_output = masks_out[0] if isinstance(masks_out, (list, tuple)) else masks_out

    # set record masks to new predicted mask matrix
    rec["masks"] = convert_cellpose_mask_to_single_array(
        mask_output, rec["H"], rec["W"]
    )
    # clear any labels in the record (no new masks are labelled)
    rec["labels"] = {
        int(i): None for i in np.unique(rec["masks"]) if i != 0
    }


# -----------------------------------------------------#
# ----------     SEGMENTATION FUNCTIONS.     --------- #
# -----------------------------------------------------#


def segment_current_and_refresh():
    """calls cellpose to segment the current image"""
    rec = get_current_rec()
    if rec is not None:
        params = get_cellpose_hparams_from_state()
        segment_with_cellpose(rec, **params)
        st.session_state["edit_canvas_nonce"] += 1
    st.rerun()


def batch_segment_and_refresh():
    """calls cellpose to segment all images with progress bar"""
    ok = ordered_keys()
    params = get_cellpose_hparams_from_state()
    n = len(ok)
    pb = st.progress(0.0, text="Starting…")
    for i, k in enumerate(ok, 1):
        segment_with_cellpose(st.session_state.images.get(k), **params)
        pb.progress(i / n, text=f"Segmented {i}/{n}")


def get_cellpose_hparams_from_state():
    """calls hparam values from session state"""
    ch1 = int(st.session_state.get("cp_ch1"))
    ch2 = int(st.session_state.get("cp_ch2"))
    diameter = st.session_state.get("cp_diameter")

    return dict(
        channels=(ch1, ch2),
        diameter=diameter,
        cellprob_threshold=float(st.session_state.get("cp_cellprob_threshold")),
        flow_threshold=float(st.session_state.get("cp_flow_threshold")),
        min_size=int(st.session_state.get("cp_min_size")),
        niter=int(st.session_state.get("cp_niter")),
    )
