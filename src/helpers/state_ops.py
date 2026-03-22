from pathlib import Path
import streamlit as st
import numpy as np


def ensure_global_state() -> None:
    """Initialize all session-state keys used across panels."""
    ss = st.session_state

    # app-level state
    ss.setdefault("images", {})  # {order_key:int -> record:dict}
    ss.setdefault("name_to_key", {})  # {filename:str -> order_key:int}
    ss.setdefault("current_key", None)  # active order_key
    ss.setdefault("next_ord", 1)  # next order_key to assign
    ss.setdefault("analysis_plots", [])
    ss.setdefault("cellpose_model_bytes", None)
    ss.setdefault("cellpose_model_name", None)
    ss.setdefault("densenet_ckpt_bytes", None)
    ss.setdefault("densenet_ckpt_name", None)
    ss.setdefault("side_new_label", "")
    ss.setdefault("show_overlay", True)
    st.session_state.setdefault("show_normalized", False)
    ss.setdefault("interaction_mode", "Remove mask")
    ss.setdefault("side_interaction_mode", "Draw box")
    ss.setdefault("skipped_files", [])
    ss.setdefault("remove_click", False)
    ss.setdefault("class_click", False)
    st.session_state.setdefault("last_class_xy", None)
    st.session_state.setdefault("last_remove_xy", None)
    ss.setdefault("disp_w", 0)

    # cellpose inference
    ss.setdefault("cp_ch1", 0)
    ss.setdefault("cp_ch2", 0)
    ss.setdefault("cp_min_size", 0)
    ss.setdefault("cp_niter", 0)
    ss.setdefault("cp_flow_threshold", 0.3)
    ss.setdefault("cp_cellprob_threshold", 0.2)
    ss.setdefault("cp_diameter", 0)

    # densenet
    ss.setdefault("densenet_model", None)

    # image dataset download options
    ss.setdefault("dl_normalize_download", False)

    # UI defaults / nonces
    ss.setdefault("pred_canvas_nonce", 0)
    ss.setdefault("edit_canvas_nonce", 0)
    ss.setdefault("mask_uploader_nonce", 0)
    ss.setdefault("image_uploader_nonce", 0)
    ss.setdefault("side_panel", "Upload data")

    # class defaults
    ss.setdefault("all_classes", ["No label"])
    ss.setdefault("side_current_class", ss["all_classes"][0])
    ss.setdefault("densenet_class_map", {})  # {pred_class_idx:int -> app_label:str}


def reset_global_state() -> None:
    """Reset ALL session_state keys to their original default values."""
    ss = st.session_state
    ss.clear()  # completely wipe current state

    # --- app-level state defaults ---
    ss["images"] = {}
    ss["name_to_key"] = {}
    ss["current_key"] = None
    ss["next_ord"] = 1
    ss["analysis_plots"] = []
    ss["cellpose_model_bytes"] = None
    ss["cellpose_model_name"] = None
    ss["densenet_ckpt_bytes"] = None
    ss["densenet_ckpt_name"] = None
    ss["side_new_label"] = ""
    ss["show_overlay"] = True
    ss["show_normalized"] = False
    ss["interaction_mode"] = "Remove mask"
    ss["side_interaction_mode"] = "Draw box"
    ss["skipped_files"] = []
    ss["remove_click"] = False
    ss["class_click"] = False
    ss["last_class_xy"] = None
    ss["last_remove_xy"] = None
    ss["disp_w"] = 0

    # --- Cellpose inference defaults ---
    ss["cp_ch1"] = 0
    ss["cp_ch2"] = 0
    ss["cp_min_size"] = 0
    ss["cp_niter"] = 500
    ss["cp_flow_threshold"] = 0.0
    ss["cp_cellprob_threshold"] = 0.0
    ss["cp_diameter"] = 0

    # --- DenseNet model ---
    ss["densenet_model"] = None

    # --- image dataset download options ---
    ss["dl_normalize_download"] = False

    # --- UI defaults / nonces ---
    ss["pred_canvas_nonce"] = 0
    ss["edit_canvas_nonce"] = 0
    ss["mask_uploader_nonce"] = 0
    ss["image_uploader_nonce"] = 0
    ss["side_panel"] = "Upload data"

    # --- class defaults ---
    ss["all_classes"] = ["No label"]
    ss["side_current_class"] = ss["all_classes"][0]


def stem(p: str) -> str:
    return Path(p).stem


def ordered_keys():
    return sorted(st.session_state.images.keys())


def get_current_rec():
    k = st.session_state.get("current_key")
    return st.session_state.images.get(k) if k is not None else None


def set_current_by_index(idx: int):
    ok = ordered_keys()
    if not ok:
        return
    st.session_state.current_key = ok[idx % len(ok)]


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalizes image intensities for Cellpose input.
    Scales mean intensity to ~127.5 or full uint8 range if mean <= 0.
    """
    im = image.astype(np.float32)
    if im.size == 0:
        return im

    mean_val = float(im.mean())
    if mean_val <= 0:
        # fallback: scale to full uint8 range
        rng = float(im.max() - im.min())
        im = (im - im.min()) / rng * 255.0 if rng > 0 else im * 0.0
    else:
        # scale by ratio so mean intensity ≈ 127.5 (mid-gray)
        im = im * (127.5 / mean_val)

    # ensure valid uint8 range
    im = np.clip(im, 0, 255)
    return im.astype(np.uint8)
