# import os, tempfile, hashlib, time
# import numpy as np
# import streamlit as st
# import cv2
# from tensorflow.keras.models import load_model

# # ---------- cache/load model once ----------
# @st.cache_resource
# def _load_densenet_model_cached_from_path(ckpt_path: str):
#     ext = os.path.splitext(ckpt_path)[1].lower()
#     if ext in (".keras", ".h5"):


#         model = load_model(ckpt_path)
#         return ("keras", model, None)
#     else:
#         import torch
#         from torchvision import models as tvm
#         import torch.nn as nn

#         device = (
#             "cuda"
#             if torch.cuda.is_available()
#             else (
#                 "mps"
#                 if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
#                 else "cpu"
#             )
#         )
#         base = tvm.densenet121(weights=None)
#         ckpt = torch.load(ckpt_path, map_location="cpu")
#         state = ckpt.get("state_dict", ckpt)
#         # infer classes if present
#         if "classifier.weight" in state:
#             num_classes = state["classifier.weight"].shape[0]
#         else:
#             num_classes = 2
#         in_feats = base.classifier.in_features
#         base.classifier = nn.Linear(in_feats, num_classes)
#         base.load_state_dict(state, strict=False)
#         base.eval().to(device)
#         return ("torch", base, device)


# def _materialize_ckpt_from_ss() -> str | None:
#     b = st.session_state.get("densenet_ckpt_bytes")
#     n = st.session_state.get("densenet_ckpt_name")
#     if not b or not n:
#         return None
#     h = hashlib.sha1(b).hexdigest()[:12]
#     path = os.path.join(
#         tempfile.gettempdir(), f"densenet_{h}{os.path.splitext(n)[1] or '.pt'}"
#     )
#     if not os.path.exists(path):
#         with open(path, "wb") as f:
#             f.write(b)
#     return path


# def _prep_batch_for_torch(gray_batch, size=224):
#     # gray_batch: list of 2D arrays
#     import torch

#     arrs = []
#     for g in gray_batch:
#         x = np.stack([g] * 3, axis=-1)
#         x = (
#             cv2.resize(x, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)
#             / 255.0
#         )
#         # ImageNet norm
#         mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
#         std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
#         x = (x - mean) / std
#         x = np.transpose(x, (2, 0, 1))  # HWC->CHW
#         arrs.append(x)
#     X = np.stack(arrs, axis=0)  # [B,3,H,W]
#     return torch.from_numpy(X)


# def _prep_batch_for_keras(gray_batch, size=224):
#     arrs = []
#     for g in gray_batch:
#         x = np.stack([g] * 3, axis=-1)
#         x = (
#             cv2.resize(x, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)
#             / 255.0
#         )
#         arrs.append(x)
#     X = np.stack(arrs, axis=0)  # [B,H,W,3]
#     return X


# def _mask_bbox_fast(mask2d: np.ndarray):
#     # quickly get tight bbox; returns None if empty
#     rows = np.any(mask2d, axis=1)
#     cols = np.any(mask2d, axis=0)
#     if not rows.any() or not cols.any():
#         return None
#     y0 = rows.argmax()
#     y1 = rows.size - rows[::-1].argmax()
#     x0 = cols.argmax()
#     x1 = cols.size - cols[::-1].argmax()
#     return y0, x0, y1, x1


# def classify_rec_with_densenet_batched(rec: dict, *, size=224, batch_size=64):
#     """
#     Classifies each mask in rec['masks'] with uploaded DenseNet-121.
#     Overwrites rec['labels'] with predicted ints; returns rec.
#     """
#     if rec is None or "image" not in rec:
#         st.warning("No current image.")
#         return rec

#     m = rec.get("masks")
#     if not isinstance(m, np.ndarray) or m.ndim != 3 or m.shape[0] == 0:
#         st.warning("No masks to classify.")
#         rec["labels"] = []
#         return rec

#     img = rec["image"]
#     if img.ndim == 3:
#         if img.shape[2] == 4:
#             img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
#         img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#     elif img.ndim != 2:
#         st.error(f"Unsupported image shape {img.shape}")
#         return rec

#     ckpt_path = _materialize_ckpt_from_ss()
#     if ckpt_path is None:
#         st.warning("Upload a DenseNet-121 checkpoint first.")
#         return rec

#     backend, model, device = _load_densenet_model_cached_from_path(ckpt_path)

#     # --- build masked crops (grayscale) ---
#     t0 = time.perf_counter()
#     crops = []
#     order = []  # keep mask index mapping
#     for i in range(m.shape[0]):
#         mask_i = m[i] > 0
#         bb = _mask_bbox_fast(mask_i)
#         if bb is None:
#             crops.append(np.zeros((size, size), dtype=np.float32))  # dummy
#             order.append(i)
#             continue
#         y0, x0, y1, x1 = bb
#         crop = img[y0:y1, x0:x1]
#         crop_mask = mask_i[y0:y1, x0:x1]
#         crop = (crop * crop_mask).astype(np.float32)
#         crops.append(crop)
#         order.append(i)
#     t_prep = time.perf_counter() - t0

#     # --- batched inference ---
#     preds = np.zeros((len(crops),), dtype=np.int64)
#     t1 = time.perf_counter()

#     if backend == "torch":
#         import torch

#         with torch.inference_mode():
#             for s in range(0, len(crops), batch_size):
#                 batch = crops[s : s + batch_size]
#                 X = _prep_batch_for_torch(batch, size=size).to(device)
#                 out = model(X)
#                 preds[s : s + batch_size] = out.argmax(dim=1).cpu().numpy()
#     else:
#         Xall = _prep_batch_for_keras(crops, size=size)
#         # use batch_size here too
#         probs = model.predict(Xall, verbose=0, batch_size=batch_size)
#         preds = np.argmax(probs, axis=-1)

#     t_infer = time.perf_counter() - t1

#     # --- write back labels in mask order ---
#     labels = [int(preds[order.index(i)]) for i in range(m.shape[0])]
#     rec["labels"] = labels

#     # optional: quick timing readout
#     st.caption(
#         f"Preprocess: {t_prep:.3f}s • Inference: {t_infer:.3f}s • Masks: {m.shape[0]}"
#     )
#     return rec
