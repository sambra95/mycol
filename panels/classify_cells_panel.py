# panels/classify_cells.py
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates
import pandas as pd
import cv2

# from helpers.densenet_functions import classify_rec_with_densenet_batched
from helpers.mask_editing_functions import (
    composite_over_by_class,
)

from helpers.state_ops import ordered_keys, set_current_by_index, current

from helpers.classifying_functions import (
    classes_map_from_labels,
    _add_label_from_input,
    classify_cells_with_densenet,
    color_hex_for,
    palette_from_emojis,
    remove_class_everywhere,
    _rename_class_from_input,
    _row,
)


# panels/classify_cells.py (sidebar only)

import numpy as np
import streamlit as st
from PIL import Image
import cv2

from helpers.state_ops import ordered_keys, set_current_by_index, current
from helpers.classifying_functions import (
    _add_label_from_input,
)

# --- fixed class-color palette (RGB hex) ---
# taken from your image (left→right), plus the gray at the end

# --- fixed class-color palette (RGB hex) ---


# --- CLASS RENAME / MERGE HELPERS ---

ss = st.session_state
ss.setdefault("all_classes", ["Remove label"])


def render_sidebar(*, key_ns: str = "side"):

    ok = ordered_keys()
    if not ok:
        st.warning("Upload an image in **Upload data** first.")
        return

    # ---- promote any class selected in previous run BEFORE widgets are created ----
    ss = st.session_state
    ss.setdefault("all_classes", ["Remove label"])
    if "pending_class" in ss:
        pc = ss.pop("pending_class")
        if pc not in ss["all_classes"]:
            ss["all_classes"].append(pc)
        ss["side_current_class"] = pc
    ss.setdefault("side_current_class", ss["all_classes"][0])

    # ---- navigation / overlay toggle ----
    rec = current()
    names = [st.session_state.images[k]["name"] for k in ok]
    i = ok.index(st.session_state.current_key)
    st.markdown(f"**Image {i+1}/{len(ok)}:** {names[i]}")

    c1, c2 = st.columns(2)
    if c1.button("◀ Prev", key=f"{key_ns}_prev", use_container_width=True):
        set_current_by_index(i - 1)
        st.rerun()
    if c2.button("Next ▶", key=f"{key_ns}_next", use_container_width=True):
        set_current_by_index(i + 1)
        st.rerun()

    st.toggle("Show mask overlay", key="show_overlay")

    st.divider()

    st.markdown("### Classify cells")

    if st.button("Classify this image with DenseNet-121", use_container_width=True):
        classify_cells_with_densenet(rec)

        # batch segment all images
    if st.button(
        "Batch classify all images with with DenseNet-121",
        key="btn_batch_classify_cellpose",
        use_container_width=True,
    ):
        n = len(ordered_keys())
        pb = st.progress(0.0, text="Starting…")
        for i, k in enumerate(ordered_keys(), 1):
            classify_cells_with_densenet(st.session_state.images.get(k))
            pb.progress(i / n, text=f"Segmented {i}/{n}")
        pb.empty()

    labels = ss.setdefault("all_classes", ["Remove label"])

    labdict = rec.get("labels", {}) if isinstance(rec.get("labels"), dict) else {}

    # Unlabel row
    _row(
        "Remove label", sum(1 for v in labdict.values() if v is None), key="use_unlabel"
    )

    # Actual classes
    for name in [c for c in labels if c != "Remove label"]:
        _row(name, sum(1 for v in labdict.values() if v == name), key=f"use_{name}")

    st.caption(f"Current click assign: **{ss.get('side_current_class','None')}**")

    if st.button(
        key="clear_labels_btn", use_container_width=True, label="Clear mask labels"
    ):
        rec["labels"] = {int(i): None for i in np.unique(rec["masks"]) if i != 0}

    st.divider()
    st.markdown("### Manage classes")

    st.text_input(
        "",
        key="side_new_label",
        placeholder="Enter a new class here",
        on_change=_add_label_from_input(labels, ss.get("side_new_label", "")),
    )

    st.text_input(
        "",
        key="delete_new_label",
        placeholder="Delete class here.",
        on_change=remove_class_everywhere(ss.get("delete_new_label", "")),
    )

    # Build options excluding the protected entry
    editable_classes = [
        c for c in st.session_state.get("all_classes", []) if c != "Remove label"
    ]
    if not editable_classes:
        st.caption("No classes yet. Add a class above first.")
    else:
        c1, c2 = st.columns([1, 2])
        with c1:
            st.selectbox(
                "Class to relabel",
                options=editable_classes,
                key=f"{key_ns}_rename_from",
            )
        with c2:
            st.text_input(
                "New label",
                key=f"{key_ns}_rename_to",
                placeholder="Type the new class name and press Enter",
                on_change=_rename_class_from_input(
                    f"{key_ns}_rename_from", f"{key_ns}_rename_to"
                ),
            )


def render_main(*, key_ns: str = "edit"):

    rec = current()
    if rec is None:
        st.warning("Upload an image in **Upload data** first.")
        return

    scale = 1.5
    H, W = rec["H"], rec["W"]
    disp_w, disp_h = int(W * scale), int(H * scale)

    M = rec.get("masks")
    has_instances = isinstance(M, np.ndarray) and M.ndim == 2 and M.any()
    if st.session_state.get("show_overlay", False) and has_instances:
        # classes actually present in this image
        classes_map = classes_map_from_labels(
            M, rec.get("labels", {})
        )  # {id -> class or None}
        present_classes = sorted(
            {c for c in classes_map.values() if c and c != "Remove label"}
        )

        # ensure each present class has a stable session color (also keeps table + overlay in sync)
        _ = [color_hex_for(c) for c in present_classes]

        # build the palette from the present classes; if none, fall back to the global list
        labels_global = st.session_state.setdefault("all_classes", ["Remove label"])
        palette = palette_from_emojis(present_classes or labels_global)

        display_for_ui = composite_over_by_class(
            rec["image"], M, classes_map, palette, alpha=0.35
        )

    else:
        display_for_ui = np.array(
            Image.fromarray(rec["image"].astype(np.uint8)).resize(
                (disp_w, disp_h), Image.BILINEAR
            )
        )

    click = streamlit_image_coordinates(
        display_for_ui, key=f"{key_ns}_img_click", width=disp_w
    )

    if click and has_instances:
        x0 = int(round(int(click["x"]) / scale))
        y0 = int(round(int(click["y"]) / scale))
        if 0 <= x0 < W and 0 <= y0 < H and (x0, y0) != rec.get("last_click_xy"):
            iid = int(M[y0, x0])
            if iid > 0:
                cur_class = st.session_state.get("side_current_class")
                if cur_class is not None:
                    if cur_class == "Remove label":
                        rec.setdefault("labels", {}).pop(iid, None)
                    else:
                        rec.setdefault("labels", {})[iid] = cur_class
                    rec["last_click_xy"] = (x0, y0)
                    st.rerun()
            else:
                rec["last_click_xy"] = (x0, y0)
