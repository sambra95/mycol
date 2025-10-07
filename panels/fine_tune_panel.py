# panels/train_densenet.py
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image  # (only if your helpers rely on PIL types somewhere)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score
from tensorflow.keras.callbacks import EarlyStopping

# ---- app helpers ----
from helpers.state_ops import ordered_keys
from helpers.densenet_functions import (
    load_labeled_patches_from_session,
    AugSequence,
    build_densenet,
    _plot_confusion_matrix,
    _plot_densenet_losses,
)
from helpers.cellpose_functions import (
    finetune_cellpose_from_records,
    _plot_losses,
    compare_models_mean_iou_plot,
)

ss = st.session_state


# ========== DenseNet: options (light) + dataset summary (light-ish) + training (heavy) ==========


def _densenet_options(key_ns="train_densenet"):
    """Light controls – lives outside fragments so changing options refreshes summary."""
    st.header("Train DenseNet on labeled cell patches")

    if not ordered_keys():
        st.info("Upload data and add labels in the other panels first.")
        return False

    with st.expander("Training options", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        ss.setdefault("dn_input_size", 64)
        ss.setdefault("dn_batch_size", 32)
        ss.setdefault("dn_base_trainable", False)
        ss.setdefault("dn_max_epoch", 100)
        ss.setdefault("dn_val_split", 0.2)

        ss["dn_input_size"] = c1.selectbox(
            "Input size",
            options=[64, 96, 128],
            index=[64, 96, 128].index(ss["dn_input_size"]),
        )
        ss["dn_batch_size"] = c2.selectbox(
            "Batch size",
            options=[8, 16, 32, 64],
            index=[8, 16, 32, 64].index(ss["dn_batch_size"]),
        )
        ss["dn_base_trainable"] = c3.checkbox(
            "Fine-tune base (unfreeze)", value=ss["dn_base_trainable"]
        )
        ss["dn_max_epoch"] = c4.number_input(
            "Max epochs",
            min_value=1,
            max_value=500,
            value=int(ss["dn_max_epoch"]),
            step=5,
            key="max_epoch_densenet_ui",
        )

        ss["dn_val_split"] = st.slider(
            "Validation split", 0.05, 0.4, float(ss["dn_val_split"]), 0.05
        )

    return True


@st.fragment
def densenet_summary_fragment():
    """Loads patches and shows a simple class frequency table (reruns when the page reruns)."""
    input_size = int(ss.get("dn_input_size", 64))

    # Load patches only for summary; heavy-ish but isolated here
    X, y, classes = load_labeled_patches_from_session(patch_size=input_size)

    # Count occurrences per class (ensure all classes present)
    counts = np.bincount(y, minlength=len(classes))
    freq_df = pd.DataFrame({"Class": list(classes), "Count": counts.astype(int)})

    st.write(f"Found {int(counts.sum())} patches across {len(classes)} classes.")
    st.table(freq_df)


@st.fragment
def densenet_train_fragment():
    """Runs the full DenseNet training pipeline when the button is clicked."""
    go = st.button("Start training", use_container_width=True, type="primary")
    if not go:
        return

    # Read options from session
    input_size = int(ss.get("dn_input_size", 64))
    batch_size = int(ss.get("dn_batch_size", 32))
    base_trainable = bool(ss.get("dn_base_trainable", False))
    epochs = int(ss.get("dn_max_epoch", 100))
    val_split = float(ss.get("dn_val_split", 0.2))

    # Load data (heavy)
    X, y, classes = load_labeled_patches_from_session(patch_size=input_size)
    if X.shape[0] < 2 or len(np.unique(y)) < 2:
        st.warning("Need at least 2 samples and 2 classes. Add more labeled cells.")
        return

    # Split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_split, stratify=y, random_state=42
    )

    # Data generators
    train_gen = AugSequence(
        X_train,
        y_train,
        batch_size=batch_size,
        num_transforms=3,
        shuffle=True,
        target_size=(input_size, input_size),
    )
    val_gen = AugSequence(
        X_val,
        y_val,
        batch_size=batch_size,
        num_transforms=1,
        shuffle=False,
        target_size=(input_size, input_size),
    )

    # Class weights
    class_weights = compute_class_weight(
        class_weight="balanced", classes=np.arange(len(classes)), y=y
    )
    class_weights_dict = {i: float(w) for i, w in enumerate(class_weights)}

    # Build model
    model = build_densenet(
        input_shape=(input_size, input_size, 3),
        num_classes=len(classes),
        base_trainable=base_trainable,
    )

    # Train
    es = EarlyStopping(
        monitor="val_loss", patience=15, restore_best_weights=True, verbose=1
    )
    with st.spinner("Training DenseNet…"):
        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs,
            callbacks=[es],
            class_weight=class_weights_dict,
            verbose=0,
        )

    # Evaluate on validation set
    Xv, yv = [], []
    for i in range(len(val_gen)):
        xb, yb = val_gen[i]
        Xv.append(xb)
        yv.append(yb)
    Xv = np.concatenate(Xv, axis=0)
    yv = np.concatenate(yv, axis=0)

    y_probs = model.predict(Xv, verbose=0)
    y_pred = np.argmax(y_probs, axis=1)

    st.info(
        "Model stored in session. You can use it immediately from the **Classify cells** panel."
    )

    acc = accuracy_score(yv, y_pred)
    prec = precision_score(yv, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(yv, y_pred, average="weighted", zero_division=0)
    metrics_dict = {"Accuracy": acc, "Precision": prec, "F1": f1}

    # Plots
    _plot_densenet_losses(
        history.history.get("loss", []),
        history.history.get("val_loss", []),
        metrics=metrics_dict,
    )
    cm = confusion_matrix(yv, y_pred, labels=np.arange(len(classes)))
    fig = _plot_confusion_matrix(cm, classes, normalize=False)
    st.pyplot(fig, use_container_width=True)

    # Persist the model in session
    ss["densenet_ckpt_bytes"] = model
    ss["densenet_ckpt_name"] = "densenet_finetuned"
    ss["densenet_model"] = model


def render_densenet_train_panel(key_ns: str = "train_densenet"):
    if not _densenet_options(key_ns):
        return
    densenet_summary_fragment()  # light-ish; recomputes only when page reruns
    densenet_train_fragment()  # heavy; runs only on button click


# ========== Cellpose: options + training ==========


def _cellpose_options(key_ns="train_cellpose"):
    st.header("Fine-tune Cellpose on your labeled data")

    if not ordered_keys():
        st.info("Upload data and label masks first.")
        return False

    with st.expander("Training options", expanded=True):
        c1, c2, c3, c4 = st.columns(4)

        # Defaults
        ss.setdefault("cp_base_model", "cyto2")
        ss.setdefault("cp_max_epoch", 100)
        ss.setdefault("cp_lr", 5e-5)
        ss.setdefault("cp_wd", 0.1)
        ss.setdefault("cp_nimg", 32)

        ss["cp_base_model"] = c1.selectbox(
            "Base model",
            options=["cyto2", "cyto", "nuclei", "scratch"],
            index=["cyto2", "cyto", "nuclei", "scratch"].index(ss["cp_base_model"]),
        )
        ss["cp_max_epoch"] = c2.number_input(
            "Max epochs", 1, 500, int(ss["cp_max_epoch"]), step=5
        )
        ss["cp_lr"] = c3.number_input(
            "Learning rate",
            min_value=1e-6,
            max_value=1e-2,
            value=float(ss["cp_lr"]),
            format="%.5f",
        )
        ss["cp_wd"] = c4.number_input(
            "Weight decay",
            min_value=0.0,
            max_value=1.0,
            value=float(ss["cp_wd"]),
            step=0.05,
        )

        ss["cp_nimg"] = st.slider("Images per epoch", 1, 128, int(ss["cp_nimg"]), 1)

    return True


@st.fragment
def cellpose_train_fragment():
    go = st.button("Start fine-tuning", use_container_width=True, type="primary")
    if not go:
        return

    recs = {k: st.session_state["images"][k] for k in ordered_keys()}
    base_model = ss.get("cp_base_model", "cyto2")
    epochs = int(ss.get("cp_max_epoch", 100))
    lr = float(ss.get("cp_lr", 5e-5))
    wd = float(ss.get("cp_wd", 0.1))
    nimg = int(ss.get("cp_nimg", 32))

    train_losses, test_losses, model_name = finetune_cellpose_from_records(
        recs,
        base_model=base_model,
        epochs=epochs,
        learning_rate=lr,
        weight_decay=wd,
        nimg_per_epoch=nimg,
    )

    st.success(f"Fine-tuning complete ✅ (model: {model_name})")

    st.session_state["train_losses"] = train_losses
    st.session_state["test_losses"] = test_losses

    _plot_losses(train_losses, test_losses)

    masks = [rec["masks"] for rec in recs.values()]
    images = [rec["image"] for rec in recs.values()]
    N = len(images)
    sample_n = min(50, N)  # only calculate the data for up to 50 image-mask pairs
    if N > sample_n:
        rng = (
            np.random.default_rng()
        )  # or np.random.default_rng(ss.get("cp_plot_seed"))
        idx = rng.choice(N, size=sample_n, replace=False)
        images = [images[i] for i in idx]
        masks = [masks[i] for i in idx]

    compare_models_mean_iou_plot(
        images,
        masks,
        base_model_name=base_model if base_model != "scratch" else "cyto2",
    )


def render_cellpose_train_panel(key_ns="train_cellpose"):
    if not _cellpose_options(key_ns):
        return
    cellpose_train_fragment()  # heavy; runs only on button click
