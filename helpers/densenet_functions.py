# --- DenseNet121 classification over masks -----------------------------------
import os, tempfile, hashlib
import numpy as np
import streamlit as st
import cv2


# Torch OR Keras depending on the uploaded file
def _materialize_densenet_ckpt_from_session() -> str | None:
    ss = st.session_state
    b = ss.get("densenet_ckpt_bytes")
    name = ss.get("densenet_ckpt_name")
    if not b or not name:
        return None
    h = hashlib.sha1(b).hexdigest()[:12]
    suffix = os.path.splitext(name)[1] or ".pt"
    path = os.path.join(tempfile.gettempdir(), f"densenet_{h}{suffix}")
    if not os.path.exists(path):
        with open(path, "wb") as f:
            f.write(b)
    return path


def _load_densenet_model_cached():
    """Returns a tuple (backend, model, device_or_None)
    backend: 'torch' or 'keras'"""
    ss = st.session_state
    ckpt_path = _materialize_densenet_ckpt_from_session()
    if ckpt_path is None:
        raise RuntimeError("No DenseNet checkpoint uploaded.")

    tag_src = ss.get("densenet_ckpt_bytes") or b""
    tag = hashlib.sha1(tag_src).hexdigest()[:12]

    # Reuse if already loaded
    if ss.get("densenet_model_obj") is not None and ss.get("densenet_model_tag") == tag:
        return (
            ss["densenet_model_backend"],
            ss["densenet_model_obj"],
            ss.get("densenet_model_device"),
        )

    ext = os.path.splitext(ckpt_path)[1].lower()

    if ext in (".keras", ".h5"):
        # Keras
        from tensorflow.keras.models import load_model

        model = load_model(ckpt_path)
        model.trainable = False
        ss["densenet_model_backend"] = "keras"
        ss["densenet_model_obj"] = model
        ss["densenet_model_tag"] = tag
        ss["densenet_model_device"] = None
        return "keras", model, None

    # Torch default
    import torch
    from torchvision import models as tvm

    device = (
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )

    # Build skeleton
    base = tvm.densenet121(weights=None)
    # Infer num_classes from checkpoint if possible
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    # accept plain state_dict or full checkpoint
    if "classifier.weight" in state:
        num_classes = state["classifier.weight"].shape[0]
    else:
        # fallback to 2-class if unknown
        num_classes = 2
    in_feats = base.classifier.in_features
    import torch.nn as nn

    base.classifier = nn.Linear(in_feats, num_classes)

    # load weights (relaxed)
    missing, unexpected = base.load_state_dict(state, strict=False)
    if len(missing) > 0 and len(unexpected) > 0:
        # best-effort; continue
        pass

    base.eval().to(device)

    ss["densenet_model_backend"] = "torch"
    ss["densenet_model_obj"] = base
    ss["densenet_model_tag"] = tag
    ss["densenet_model_device"] = device
    return "torch", base, device


def _prep_crop_for_torch(crop_gray: np.ndarray, size=224):
    # stack to 3ch, resize, to [0,1], normalize ImageNet, to CHW tensor
    import torch

    crop = np.stack([crop_gray] * 3, axis=-1)
    crop = (
        cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)
        / 255.0
    )
    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    crop = (crop - mean) / std
    crop = np.transpose(crop, (2, 0, 1))  # HWC -> CHW
    return torch.from_numpy(crop).unsqueeze(0)  # [1,3,H,W]


def _prep_crop_for_keras(crop_gray: np.ndarray, size=224):
    crop = np.stack([crop_gray] * 3, axis=-1)
    crop = (
        cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)
        / 255.0
    )
    return np.expand_dims(crop, axis=0)  # [1,H,W,3]


def _mask_bbox(mask2d: np.ndarray):
    ys, xs = np.where(mask2d > 0)
    if ys.size == 0:
        return None
    return ys.min(), xs.min(), ys.max() + 1, xs.max() + 1


def classify_rec_with_densenet(rec: dict, *, size=224):
    """
    Uses uploaded DenseNet-121 to classify each mask in rec['masks'].
    Overwrites rec['labels'] with predicted ints (argmax per mask).
    """
    if rec is None or "image" not in rec:
        st.warning("No current image to classify.")
        return rec

    m = rec.get("masks")
    if not isinstance(m, np.ndarray) or m.ndim != 3 or m.shape[0] == 0:
        st.warning("No masks found to classify.")
        rec["labels"] = []
        return rec

    img = rec["image"]
    if img.ndim == 3:
        # convert to grayscale
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    elif img.ndim != 2:
        st.error(f"Unsupported image shape {img.shape}")
        return rec

    backend, model, device = _load_densenet_model_cached()

    preds = []
    if backend == "torch":
        import torch

        with torch.inference_mode():
            for i in range(m.shape[0]):
                mask_i = (m[i] > 0).astype(np.uint8)
                bb = _mask_bbox(mask_i)
                if bb is None:
                    preds.append(0)
                    continue
                y0, x0, y1, x1 = bb
                crop = img[y0:y1, x0:x1]
                # apply mask to crop region
                crop_mask = mask_i[y0:y1, x0:x1]
                crop_mul = (crop * (crop_mask > 0)).astype(np.float32)
                inp = _prep_crop_for_torch(crop_mul, size=size).to(device)
                out = model(inp)
                cls = int(out.argmax(dim=1).item())
                preds.append(cls)
    else:
        # Keras
        for i in range(m.shape[0]):
            mask_i = (m[i] > 0).astype(np.uint8)
            bb = _mask_bbox(mask_i)
            if bb is None:
                preds.append(0)
                continue
            y0, x0, y1, x1 = bb
            crop = img[y0:y1, x0:x1]
            crop_mask = mask_i[y0:y1, x0:x1]
            crop_mul = (crop * (crop_mask > 0)).astype(np.float32)
            inp = _prep_crop_for_keras(crop_mul, size=size)
            prob = model.predict(inp, verbose=0)
            cls = int(np.argmax(prob, axis=-1)[0])
            preds.append(cls)

    # overwrite labels (list of ints)
    rec["labels"] = preds
