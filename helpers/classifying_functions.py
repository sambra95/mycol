# panels/classify_cells.py
import numpy as np
import streamlit as st
from PIL import Image
import io
import pandas as pd
from pathlib import Path
from zipfile import ZipFile
import cv2
from tensorflow.keras.applications.densenet import preprocess_input
import numpy as np, cv2
from skimage.exposure import rescale_intensity
from streamlit_image_coordinates import streamlit_image_coordinates


from helpers.state_ops import ordered_keys, current

ss = st.session_state

PALETTE_HEX = [
    "#DC050C",  # 26
    "#5289C7",  # 12
    "#4EB265",  # 15
    "#F7F056",  # 18
    "#882E72",  # 9
    "#E8601C",  # 24
    "#D1BBD7",  # 3
    "#90C987",  # 16
    "#F1932D",  # 22
    "#CAE0AB",  # 17
    "#F6C141",  # 20
]
_DEFAULT_FALLBACK_HEX = "#777777"


def _hex_to_rgb01(hx: str) -> tuple[float, float, float]:
    return tuple(int(hx[i : i + 2], 16) / 255.0 for i in (1, 3, 5))


def color_hex_for(name: str) -> str:
    """
    Deterministic, unique color per class name using session-backed mapping.
    Reuses the palette without hashing collisions.
    """
    if not name or name == "Remove label":
        return _DEFAULT_FALLBACK_HEX
    cmap = st.session_state.setdefault("class_colors", {})
    if name not in cmap:
        # pick the first palette color not currently used; wrap if all are used
        used = set(cmap.values())
        choice = next((hx for hx in PALETTE_HEX if hx not in used), None)
        if choice is None:
            choice = PALETTE_HEX[len(cmap) % len(PALETTE_HEX)]
        cmap[name] = choice
    return cmap[name]


def normalize_crop(image: np.ndarray) -> np.ndarray:
    """
    Apply Cellpose-specific normalization:
    - Normalize by mean intensity
    - Rescale to 0-1
    """
    im = image.astype(np.float32)
    mean_val = np.mean(im)
    if mean_val == 0:
        raise ValueError("Image mean is zero; cannot normalize.")
    meannorm = im * (0.5 / mean_val)
    return rescale_intensity(meannorm, in_range="image", out_range=(0, 1))


def _color_chip_md(hex_color: str, size: int = 14) -> str:
    return (
        f'<span style="display:inline-block;'
        f"width:{size}px;height:{size}px;margin-top:2px;"
        f"background:{hex_color};border:1px solid rgba(0,0,0,.15);"
        f'border-radius:3px;"></span>'
    )


def _rename_class_from_input(old_key: str, new_key: str):
    """Callback: read selected old class and typed new class from session_state and rename."""

    def _cb():
        ss = st.session_state
        old = ss.get(old_key)
        new = (ss.get(new_key, "") or "").strip()
        if not old or not new or old == new:
            return

        # Keep the selection stable across reruns:
        # set the selectbox's value to the *new* name so Streamlit won't
        # fall back to the first option when `old` disappears from options.
        ss[old_key] = new
        ss[new_key] = ""

        # Validate + apply (this may call st.rerun() internally)
        rename_class_everywhere(old, new)

    return _cb


def _all_image_records():
    """Yield all image records from session state safely."""
    ims = st.session_state.get("images", {}) or {}
    for k in ordered_keys():
        rec = ims.get(k)
        if isinstance(rec, dict):
            yield k, rec


def rename_class_everywhere(old_name: str, new_name: str):
    """
    Rename a class globally across the session. If new_name already exists,
    labels with old_name are reassigned to new_name and old_name is removed.
    This updates:
      - st.session_state['all_classes']
      - per-image rec['labels'] dicts
      - st.session_state['side_current_class'] if it was old_name
      - emoji map st.session_state['class_emojis']
    """
    ss = st.session_state
    if not old_name or old_name == "Remove label":
        st.warning("That class cannot be renamed.")
        return
    new_name = (new_name or "").strip()
    if not new_name or new_name == "Remove label":
        st.warning("Please choose a non-empty class name that isn't reserved.")
        return
    if new_name == old_name:
        st.info("No change needed.")
        return

    # Ensure class list exists
    all_classes = ss.setdefault("all_classes", ["Remove label"])

    # Track whether target already exists (merge)
    target_exists = new_name in all_classes

    # --- Update labels in every image record ---
    changed_labels = 0
    for _, rec in _all_image_records():
        lab = rec.get("labels")
        if isinstance(lab, dict):
            # Re-assign values from old_name -> new_name
            to_update = [iid for iid, cname in lab.items() if cname == old_name]
            for iid in to_update:
                lab[iid] = new_name
            changed_labels += len(to_update)

    # --- Update class list (merge or rename) ---
    if old_name in all_classes:
        all_classes = [c for c in all_classes if c != old_name]  # drop old
    if new_name not in all_classes:
        all_classes.append(new_name)
    # Keep "Remove label" first if you prefer; otherwise keep as-is
    # Re-assign back
    ss["all_classes"] = all_classes

    # --- Update current selection if needed ---
    if ss.get("side_current_class") == old_name:
        ss["side_current_class"] = new_name

    # --- Update emojis ---
    emap = ss.setdefault("class_emojis", {})
    if target_exists:
        # We’re merging into an existing class; keep the target’s emoji,
        # drop the old one if present.
        if old_name in emap:
            emap.pop(old_name, None)
    else:
        # True rename: carry over old emoji if present, otherwise assign fresh later
        if old_name in emap:
            emap[new_name] = emap.pop(old_name)
        else:
            # touch to ensure emoji assignment exists for the new name
            _ = color_hex_for(new_name)

    st.rerun()


def remove_class_everywhere(name: str):
    """
    Delete a class from the session. All masks with this class are unlabelled (set to None).
    Updates:
      - st.session_state['all_classes'] (removes the class)
      - per-image rec['labels'] values (name -> None)
      - st.session_state['side_current_class'] (fallback to "Remove label" if needed)
      - emoji map st.session_state['class_emojis'] (removes entry)
    """
    ss = st.session_state

    # Ensure class list exists and cant remove "Remove label"
    if name == "Remove label":
        return
    all_classes = ss.setdefault("all_classes", ["Remove label"])
    if name not in all_classes:
        return

    # Unlabel everywhere
    changed = 0
    for _, rec in _all_image_records():
        lab = rec.get("labels")
        if isinstance(lab, dict):
            to_update = [iid for iid, cname in lab.items() if cname == name]
            for iid in to_update:
                lab[iid] = None
            changed += len(to_update)

    # Remove from class list & emoji map
    ss["all_classes"] = [c for c in all_classes if c != name]
    ss.setdefault("class_emojis", {}).pop(name, None)

    # Fix current selection if needed
    if ss.get("side_current_class") == name:
        ss["side_current_class"] = "Remove label"
    st.rerun()


def palette_from_emojis(class_names):
    """
    Return {class_name: (r,g,b)} in 0..1 using the fixed palette.
    Includes '__unlabeled__' as white.
    """
    pal = {"__unlabeled__": (1.0, 1.0, 1.0)}
    for n in class_names:
        if not n or n == "Remove label":
            continue
        pal[n] = _hex_to_rgb01(color_hex_for(n))
    return pal


def classes_map_from_labels(masks, labels):
    inst = np.asarray(masks)
    if inst.ndim != 2 or inst.size == 0:
        return {}
    classes_map = {}
    for iid in np.unique(inst):
        if iid == 0:
            continue
        cls = labels.get(int(iid))
        classes_map[int(iid)] = cls if cls not in (None, "") else "Remove label"
    return classes_map


def _stem(name: str) -> str:
    return Path(name).stem


def extract_masked_cell_patch(
    image: np.ndarray, mask: np.ndarray, size: int | tuple[int, int] = 64
):
    im, m = np.asarray(image), np.asarray(mask, bool)
    if im.shape[:2] != m.shape:
        raise ValueError("image/mask size mismatch")
    if not m.any():
        return None
    if im.ndim == 3 and im.shape[2] == 4:
        im = cv2.cvtColor(im, cv2.COLOR_RGBA2RGB)

    ys, xs = np.where(m)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    crop, mc = im[y0:y1, x0:x1], m[y0:y1, x0:x1]
    crop = (crop * mc[..., None] if crop.ndim == 3 else crop * mc).astype(im.dtype)

    tw, th = (size, size) if isinstance(size, int) else map(int, size)
    h, w = crop.shape[:2]
    s = min(tw / w, th / h)
    nw, nh = max(1, int(w * s)), max(1, int(h * s))
    resized = cv2.resize(
        crop, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR
    )

    canvas = np.zeros(
        (th, tw) if resized.ndim == 2 else (th, tw, resized.shape[2]), dtype=im.dtype
    )  # black pad
    yx = ((th - nh) // 2, (tw - nw) // 2)
    canvas[yx[0] : yx[0] + nh, yx[1] : yx[1] + nw, ...] = resized
    return canvas


def _u8(a):
    if a.dtype == np.uint8:
        return a
    a = a.astype(np.float32)
    return np.clip(a * 255 if a.max() <= 1 else a, 0, 255).astype(np.uint8)


def make_classifier_zip(patch_size: int = 64) -> bytes | None:
    rows, buf = [], io.BytesIO()
    with ZipFile(buf, "w") as zf:
        for k in ordered_keys():
            rec = st.session_state.images[k]
            img, M = rec.get("image"), rec.get("masks")
            if (
                img is None
                or not isinstance(M, np.ndarray)
                or M.ndim != 3
                or M.shape[0] == 0
            ):
                continue
            M = (M > 0).astype(np.uint8)
            labs = list(rec.get("labels", [])) + [None] * max(
                0, M.shape[0] - len(rec.get("labels", []))
            )
            inst = stack_to_instances_binary_first(M)
            base = _stem(rec["name"])
            for iid in np.unique(inst)[1:]:
                mm = (inst == int(iid)).astype(np.uint8)
                if not mm.any():
                    continue
                owner = int(np.argmax(M[:, mm.astype(bool)].sum(axis=1)))
                cls = labs[owner]
                if cls in (None, "Remove label"):
                    continue
                patch = extract_masked_cell_patch(img, mm, size=patch_size)
                if patch is None:
                    continue
                name = f"{base}_mask{int(iid)}.png"
                bio = io.BytesIO()
                Image.fromarray(_u8(patch)).save(bio, "PNG")
                zf.writestr(f"images/{name}", bio.getvalue())
                rows.append({"image": name, "mask number": int(iid), "class": cls})
        if rows:
            zf.writestr("labels.csv", pd.DataFrame(rows).to_csv(index=False))
    return buf.getvalue() if rows else None


def _row(name: str, count: int, key: str, mode_ns: str = "side"):
    # icon | name | count | select |
    c1, c2, c3, c4 = st.columns([1, 5, 2, 3])
    if name == "Remove label":
        c1.write(" ")
    else:
        c1.markdown(_color_chip_md(color_hex_for(name)), unsafe_allow_html=True)
    c2.write(f"**{name}**")
    c3.write(str(count))

    def _select():
        # pick this class AND switch the main panel to Assign class mode
        st.session_state["pending_class"] = name
        st.session_state[f"interaction_mode"] = "Assign class"

    c4.button(
        "Select",
        key=f"{key}_select",
        use_container_width=True,
        on_click=_select,
    )


def _add_label_from_input(labels, new_label_ss):
    new_label = new_label_ss.strip()
    if not new_label:
        return
    # assumes `labels` is in scope; consider storing it in st.session_state if needed
    if new_label not in labels:
        labels.append(new_label)
    st.session_state["side_current_class"] = new_label
    st.session_state["side_new_label"] = ""


def resize_with_aspect_ratio(img, target_size=64):
    h, w = img.shape[:2]
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # create square canvas
    canvas = np.zeros(
        (target_size, target_size, *([img.shape[2]] if img.ndim == 3 else [])),
        dtype=img.dtype,
    )
    y0 = (target_size - new_h) // 2
    x0 = (target_size - new_w) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas


def classify_cells_with_densenet(rec: dict) -> None:
    """Classify segmented cell masks in `rec` using a DenseNet-121 model.
    Mutates `rec` and session_state, then triggers a rerun on success.
    """
    model = ss.get("densenet_model")
    if model is None:
        st.warning("Upload a DenseNet-121 classifier in the sidebar first.")
        return

    M = rec.get("masks")
    if not isinstance(M, np.ndarray) or M.ndim != 2 or not np.any(M):
        st.info("No masks to classify.")
        return

    # Build usable class names (fallback to two defaults)
    all_classes = [c for c in ss.get("all_classes", []) if c != "Remove label"] or [
        "class0",
        "class1",
    ]

    # Gather instance ids and extract 64x64 patches
    ids = [int(v) for v in np.unique(M) if v != 0]
    patches, keep_ids = [], []

    for iid in ids:
        patch = np.asarray(extract_masked_cell_patch(rec["image"], M == iid, size=64))
        if patch.ndim == 2:
            patch = np.repeat(patch[..., None], 3, axis=2)
        elif patch.ndim == 3 and patch.shape[2] == 4:
            patch = cv2.cvtColor(patch, cv2.COLOR_RGBA2RGB)
        elif patch.ndim == 3 and patch.shape[2] == 3:
            patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)

        patch = resize_with_aspect_ratio(patch, 64)
        patches.append(preprocess_input(patch.astype(np.float32)))
        keep_ids.append(iid)

    if not patches:
        st.info("No valid patches extracted.")
        return

    # Predict classes
    X = np.stack(patches, axis=0)
    preds = model.predict(X, verbose=0).argmax(axis=1)

    # Write back labels, extend class list if needed
    labels = rec.setdefault("labels", {})
    for iid, cls_idx in zip(keep_ids, preds):
        idx = int(cls_idx)
        name = all_classes[idx] if idx < len(all_classes) else str(idx)
        labels[int(iid)] = name
        if name and name != "Remove label" and name not in ss.get("all_classes", []):
            ss.setdefault("all_classes", []).append(name)

    # Persist updated record and rerun UI
    ss.images[ss.current_key] = rec


def stack_to_instances_binary_first(m: np.ndarray) -> np.ndarray:
    """
    Robust stack (N,H,W,[...]) -> instance labels (H,W) uint16.
    - Treats >0 as 1
    - Resolves overlaps by descending area (largest wins), matching UI behavior.
    """
    m = np.asarray(m)
    if m.ndim == 4 and m.shape[-1] in (1, 3):
        m = m[..., 0]
    if m.ndim == 3 and m.shape[-1] in (1, 3):  # (H,W,1) sneaking in as 'stack'
        m = m[..., 0][None, ...]
    if m.ndim == 2:
        m = m[None, ...]
    bin_stack = (m > 0).astype(np.uint8)

    N, H, W = bin_stack.shape
    areas = bin_stack.reshape(N, -1).sum(axis=1)
    order = np.argsort(-areas)  # largest first

    inst = np.zeros((H, W), dtype=np.uint16)
    curr_id = 1
    for ch in order:
        mm = bin_stack[ch] > 0
        if mm.sum() == 0:
            continue
        write_here = mm & (inst == 0)
        if write_here.any():
            inst[write_here] = curr_id
            curr_id += 1
    return inst


# ---------- Small utilities ----------


def _ensure_defaults():
    ss = st.session_state
    ss.setdefault("all_classes", ["Remove label"])
    ss.setdefault("side_current_class", ss["all_classes"][0])


def _image_display(rec, scale):
    """Return a resized UI image honoring the overlay toggle + dims."""
    H, W = rec["H"], rec["W"]
    disp_w, disp_h = int(W * scale), int(H * scale)

    M = rec.get("masks")
    has_instances = isinstance(M, np.ndarray) and M.ndim == 2 and M.any()

    if st.session_state.get("show_overlay", False) and has_instances:
        # classes present in this image
        classes_map = classes_map_from_labels(M, rec.get("labels", {}))
        present_classes = sorted(
            {c for c in classes_map.values() if c and c != "Remove label"}
        )
        # keep a stable session color for each present class
        _ = [color_hex_for(c) for c in present_classes]
        # palette from present classes; fall back to global list
        labels_global = st.session_state.setdefault("all_classes", ["Remove label"])
        palette = palette_from_emojis(present_classes or labels_global)
        base_img = composite_over_by_class(
            rec["image"], M, classes_map, palette, alpha=0.35
        )
    else:
        base_img = rec["image"]

    display_for_ui = np.array(
        Image.fromarray(base_img.astype(np.uint8)).resize(
            (disp_w, disp_h), Image.BILINEAR
        )
    )
    return display_for_ui, disp_w, disp_h


# ---------- Sidebar: navigation / overlay toggle (simple, full rerun) ----------


def nav_fragment(key_ns="side"):
    ok = ordered_keys()
    if not ok:
        st.warning("Upload an image in **Upload data** first.")
        return

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


# ---------- Sidebar: actions (heavy) ----------


@st.fragment
def classify_actions_fragment():
    rec = current()
    st.button(
        "Classify this image with DenseNet-121",
        use_container_width=True,
        on_click=lambda: _classify_one_and_refresh(rec),
    )

    st.button(
        "Batch classify all images with DenseNet-121",
        key="btn_batch_classify_cellpose",
        use_container_width=True,
        on_click=_batch_classify_and_refresh,
    )


def _classify_one_and_refresh(rec):
    if rec is not None:
        classify_cells_with_densenet(rec)
    st.rerun()


def _batch_classify_and_refresh():
    ok = ordered_keys()
    if not ok:
        return
    n = len(ok)
    pb = st.progress(0.0, text="Starting…")
    for i, k in enumerate(ok, 1):
        classify_cells_with_densenet(st.session_state.images.get(k))
        pb.progress(i / n, text=f"Classified {i}/{n}")
    pb.empty()
    st.rerun()


# ---------- Sidebar: choose current class + clear labels (light) ----------


@st.fragment
def class_selection_fragment():
    _ensure_defaults()

    # Promote any pending class BEFORE widgets are created
    ss = st.session_state
    if "pending_class" in ss:
        pc = ss.pop("pending_class")
        if pc not in ss["all_classes"]:
            ss["all_classes"].append(pc)
        ss["side_current_class"] = pc
    ss.setdefault("side_current_class", ss["all_classes"][0])

    rec = current()
    labels = ss.setdefault("all_classes", ["Remove label"])
    labdict = rec.get("labels", {}) if isinstance(rec.get("labels"), dict) else {}

    # Unlabel row
    _row(
        "Remove label", sum(1 for v in labdict.values() if v is None), key="use_unlabel"
    )

    # Actual classes
    for name in [c for c in labels if c != "Remove label"]:
        _row(name, sum(1 for v in labdict.values() if v == name), key=f"use_{name}")

    if st.button(
        key="clear_labels_btn", use_container_width=True, label="Clear mask labels"
    ):
        rec["labels"] = {int(i): None for i in np.unique(rec["masks"]) if i != 0}
        st.rerun()


# ---------- Sidebar: manage classes (light) ----------


@st.fragment
def class_manage_fragment(key_ns="side"):
    ss = st.session_state
    labels = ss.setdefault("all_classes", ["Remove label"])

    st.markdown("### Add and remove classes")
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

    editable = [c for c in ss.get("all_classes", []) if c != "Remove label"]
    if not editable:
        st.caption("No classes yet. Add a class above first.")
        return

    c1, c2 = st.columns([1, 2])
    with c1:
        st.selectbox("Class to relabel", options=editable, key=f"{key_ns}_rename_from")
    with c2:
        st.text_input(
            "New label",
            key=f"{key_ns}_rename_to",
            placeholder="Type the new class name and press Enter",
            on_change=_rename_class_from_input(
                f"{key_ns}_rename_from", f"{key_ns}_rename_to"
            ),
        )


# ---------- Main: display ----------


@st.fragment
def display_and_click_fragment(*, key_ns="edit", scale=1.5):
    _ensure_defaults()
    rec = current()
    if rec is None:
        st.warning("Upload an image in **Upload data** first.")
        return

    H, W = rec["H"], rec["W"]
    display_for_ui, disp_w, disp_h = _image_display(rec, scale)

    M = rec.get("masks")
    has_instances = isinstance(M, np.ndarray) and M.ndim == 2 and M.any()

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
