# helpers/densenet_functions.py
import numpy as np
import streamlit as st
import cv2

import torch
import torch.nn as nn
from torchvision import models

from src.helpers.state_ops import normalize_image

ss = st.session_state


# -------------------------------
#  Device Configuration
# -------------------------------
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# -------------------------------
#  Preprocessing and loader functions
# -------------------------------


def generate_cell_patch(image: np.ndarray, mask: np.ndarray, patch_size: int = 64):
    """takes an image and boolean mask input and a normalized square patch image from the mask"""
    # extract bounding box crop
    im, m = np.asarray(image), np.asarray(mask, bool)

    # handle empty mask case
    ys, xs = np.where(m)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    crop, mc = im[y0:y1, x0:x1], m[y0:y1, x0:x1]
    crop = (crop * mc[..., None] if crop.ndim == 3 else crop * mc).astype(im.dtype)

    # checks to make sure crop is the correct format
    if crop.ndim == 2:
        crop = np.stack([crop] * 3, axis=-1)
    elif crop.ndim == 3 and crop.shape[2] == 4:
        crop = cv2.cvtColor(crop, cv2.COLOR_RGBA2RGB)
    elif crop.ndim == 3 and crop.shape[2] == 3:
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    else:
        crop = crop[..., :3]

    # resize to patch size
    crop = resize_with_aspect_ratio(crop, patch_size=patch_size)
    return crop.astype(np.float32)


def resize_with_aspect_ratio(img: np.ndarray, patch_size=64) -> np.ndarray:
    """resizes input image to a square with 'patch_size' height while maintaining the aspect ratio"""
    th, tw = patch_size, patch_size
    h, w = img.shape[:2]

    # resize with aspect ratio
    scale = min(th / h, tw / w)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(
        img, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    )

    # pad to target size
    if img.ndim == 2:
        canvas = np.zeros((th, tw), dtype=img.dtype)
        y0, x0 = (th - nh) // 2, (tw - nw) // 2
        canvas[y0 : y0 + nh, x0 : x0 + nw] = resized
    else:
        c = img.shape[2]
        canvas = np.zeros((th, tw, c), dtype=img.dtype)
        y0, x0 = (th - nh) // 2, (tw - nw) // 2
        canvas[y0 : y0 + nh, x0 : x0 + nw, :] = resized

    return canvas


def generate_patches_with_ids(rec, patch_size=64):
    """returns list of cell patches and patch ids from input record"""
    M = rec.get("masks")
    # extract the individual masks
    ids = [int(v) for v in np.unique(M) if v != 0]

    patches, keep_ids = [], []
    for iid in ids:
        patches.append(
            generate_cell_patch(
                image=rec["image"], mask=M == iid, patch_size=patch_size
            )
        )
        keep_ids.append(iid)

    return patches, keep_ids


# -------------------------------
#  Model Helper functions
# -------------------------------


def get_densenet_num_classes(model) -> int | None:
    """Infer number of output classes from the DenseNet model."""
    if model is None:
        return None
    try:
        if isinstance(model.classifier, nn.Sequential):
            last_layer = model.classifier[-1]
            return last_layer.out_features
        return model.classifier.out_features
    except Exception:
        return None


def ensure_densenet_class_map() -> dict[int, str | None]:
    """Ensure we have a mapping for each model class index in session_state."""
    ss = st.session_state
    model = ss.get("densenet_model")
    n_classes = get_densenet_num_classes(model)
    if n_classes is None:
        return {}

    class_map = ss.setdefault("densenet_class_map", {})
    # Make sure there is a key for each model output index
    for idx in range(n_classes):
        class_map.setdefault(idx, None)
    ss["densenet_class_map"] = class_map
    return class_map


def densenet_mapping_fragment():
    ss = st.session_state
    model = ss.get("densenet_model")
    if model is None:
        return

    n_classes = get_densenet_num_classes(model)
    all_classes = ss.setdefault("all_classes", ["No label"])
    class_map = ensure_densenet_class_map()

    for idx in range(n_classes):
        current = class_map.get(idx)
        options = all_classes
        if current in options:
            default_idx = options.index(current)
        else:
            default_idx = options.index("No label") if "No label" in options else 0

        selected = st.selectbox(
            label=f"Map model class {idx+1} to",
            options=options,
            index=default_idx,
            key=f"densenet_map_{idx}",
        )
        class_map[idx] = selected

    ss["densenet_class_map"] = class_map


def classify_cells_with_densenet(rec: dict) -> None:
    """Classify segmented cell masks in `rec` using a DenseNet-121 model."""
    ss = st.session_state
    model = ss.get("densenet_model")
    M = rec.get("masks")

    if not np.any(M) or model is None:
        return

    device = get_device()
    model.to(device)
    model.eval()

    patches, keep_ids = generate_patches_with_ids(rec)

    patches_np = [normalize_image(patch) for patch in patches]

    X_list = []
    for p in patches_np:
        p_chw = np.transpose(p, (2, 0, 1))
        X_list.append(torch.tensor(p_chw, dtype=torch.float32))

    if not X_list:
        return

    X_batch = torch.stack(X_list).to(device)

    with torch.no_grad():
        outputs = model(X_batch)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()

    class_map = ensure_densenet_class_map()
    all_classes = ss.setdefault("all_classes", ["No label"])
    labels = rec.setdefault("labels", {})

    for iid, cls_idx in zip(keep_ids, preds):
        idx = int(cls_idx)
        name = class_map.get(idx)
        if not name:
            name = "No label"
        labels[int(iid)] = name

        if name and name != "No label" and name not in all_classes:
            all_classes.append(name)

    ss["all_classes"] = all_classes


# -------------------------------
#  Densenet121 Model
# -------------------------------


def build_densenet(num_classes=2):
    model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)

    for param in model.features.parameters():
        param.requires_grad = False

    in_features = model.classifier.in_features

    model.classifier = nn.Sequential(
        nn.Linear(in_features, 128), nn.ReLU(), nn.Linear(128, num_classes)
    )
    return model
