import streamlit as st
from helpers.cellpose_functions import _has_cellpose_model
import itertools
from pathlib import Path
from cellpose import core, io, metrics, models, train


from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
from cellpose import core, io, models


from typing import Any, Dict, Iterable, List, Tuple, Union
import numpy as np


def _stack_to_label(masks: np.ndarray) -> np.ndarray:
    """
    (N,H,W) or (H,W) -> (H,W) integer labels, 0=bg, 1..N instances.
    """
    masks = np.asarray(masks)
    if masks.ndim == 2:
        return masks.astype(np.int32, copy=False)
    if masks.ndim != 3:
        raise ValueError(
            f"Unexpected masks shape {masks.shape}; expected (N,H,W) or (H,W)"
        )
    n, h, w = masks.shape
    lbl = np.zeros((h, w), dtype=np.int32)
    for i in range(n):
        m = masks[i].astype(bool, copy=False)
        lbl[m] = i + 1
    return lbl


def _sorted_items(
    recs_map: Dict[Any, Dict[str, Any]],
) -> Iterable[Tuple[Any, Dict[str, Any]]]:
    """
    Sort keys numerically when possible, else lexicographically as strings.
    """

    def _key(k: Any):
        try:
            return (0, int(k))
        except Exception:
            return (1, str(k))

    return sorted(recs_map.items(), key=lambda kv: _key(kv[0]))


def _recs_to_lists(
    recs: Union[Dict[Any, Dict[str, Any]], List[Dict[str, Any]]],
) -> Tuple[List[np.ndarray], List[np.ndarray], List[str]]:
    """
    Accepts either:
      - dict: {order_key -> record_dict}, e.g. ss["images"]
      - list: [record_dict, ...]
    Returns:
      images: list[np.ndarray]  # (H,W) or (H,W,3)
      labels: list[np.ndarray]  # (H,W) int32 label image
      files : list[str]         # pseudo-filenames for logging/compat
    """
    images: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    files: List[str] = []

    if isinstance(recs, dict):
        iterable = _sorted_items(recs)  # yields (order_key, rec)
    else:
        iterable = enumerate(recs)  # yields (i, rec)

    for idx, rec in iterable:
        if not isinstance(rec, dict):
            continue
        if "image" not in rec or rec["image"] is None:
            continue

        img = np.asarray(rec["image"])
        if img.ndim not in (2, 3):
            raise ValueError(f"rec[{idx}] image has invalid shape {img.shape}")

        # Prefer a single 2D label map; otherwise build from a stack of masks
        if "mask" in rec and rec["mask"] is not None:
            lbl = np.asarray(rec["mask"])
            if lbl.ndim != 2:
                raise ValueError(f"rec[{idx}] 'mask' must be 2D, got {lbl.shape}")
            lbl = lbl.astype(np.int32, copy=False)
        elif "masks" in rec and rec["masks"] is not None:
            lbl = _stack_to_label(np.asarray(rec["masks"]))
        else:
            # Skip unlabeled examples for fine-tuning
            continue

        images.append(img)
        # ensure label dtype matches Cellpose expectations
        labels.append(lbl.astype(np.int32, copy=False))

        name = rec.get("name")
        if not name:
            name = f"rec_{idx}.png"
        files.append(str(name))

    return images, labels, files


def split_recs_80_20(recs: dict, seed: int = 0):
    """Return (train_dict, test_dict) from a recs mapping."""
    keys = list(recs)
    if not keys:
        return {}, {}
    rng = np.random.default_rng(seed)
    rng.shuffle(keys)
    n_test = max(1, int(len(keys) * 0.2)) if len(keys) > 1 else 0
    test_keys = set(keys[:n_test])
    train = {k: recs[k] for k in keys if k not in test_keys}
    test = {k: recs[k] for k in keys if k in test_keys}
    return train, test


def train_cellpose_model(recs, init_model) -> None:
    """
    Fine-tune Cellpose using in-memory records (recs) instead of train/test folders.

    Expects each rec to contain:
      - 'image': np.ndarray (H,W) or (H,W,3)
      - EITHER 'mask': 2D label image (H,W) with 0=bg, 1..N instances
        OR    'masks': stack of binary masks (N,H,W) to be combined.
      - optional 'name': str (used for logging)
    """

    recs_train, recs_test = split_recs_80_20(recs)

    use_gpu = core.use_gpu()

    # Determine channels (grayscale training)
    # If you need RGB, adjust to [2,3] per Cellpose docs
    chan1, chan2 = "Grayscale", None
    channels = [0, 0]

    # Initial model (None => train from scratch)
    init_model = None if init_model == "Scratch" else init_model

    print(f"GPU activated: {use_gpu}")
    _ = io.logger_setup()
    cell_model = models.CellposeModel(
        gpu=use_gpu, pretrained_model="/Users/sambra/Downloads/cpsam"
    )

    # --- Build data from recs (same structure as io.load_train_test_data) ---
    train_data, train_labels, train_files = _recs_to_lists(recs_train)
    test_data, test_labels, test_files = _recs_to_lists(recs_test)

    new_path, train_losses, test_losses = train.train_seg(
        cell_model.net,
        train_data=train_data,
        train_labels=train_labels,
        test_data=test_data,
        test_labels=test_labels,
        channels=channels,
        n_epochs=100,
        learning_rate=0.1,
        weight_decay=0.001,
        SGD=True,
        nimg_per_epoch=8,
        model_name="test_finetuned.pt",
        save_path="/Users/sambra/Downloads",
        # normalize = params
    )

    return cell_model


def render_sidebar():

    if st.button(
        "Fine tune Cellpose on uploaded data",
        key="btn_segment_cellpose",
        use_container_width=True,
        help="Check all desired cells have masks before training.",
    ):

        train_cellpose_model(
            recs=st.session_state["images"],
            init_model=st.session_state["cyto_to_train"],
        )


def render_main():
    None
