# panels/train_densenet.py
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd

# ---- bring in existing app helpers ----
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


def render_densenet_train_panel(key_ns: str = "train_densenet"):
    st.header("Train DenseNet on labeled cell patches")
    if not ordered_keys():
        st.info("Upload data and add labels in the other panels first.")
        return

    with st.expander("Training options", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        input_size = c1.selectbox("Input size", options=[64, 96, 128], index=0)
        batch_size = c2.selectbox("Batch size", options=[8, 16, 32, 64], index=2)
        base_trainable = c3.checkbox("Fine-tune base (unfreeze)", value=False)
        epochs = c4.number_input(
            "Max epochs",
            min_value=1,
            max_value=500,
            value=100,
            step=5,
            key="max_epoch_densenet",
        )
        val_split = st.slider("Validation split", 0.05, 0.4, 0.2, 0.05)

    # Load patches from session
    X, y, classes = load_labeled_patches_from_session(patch_size=input_size)
    if X.shape[0] < 2 or len(np.unique(y)) < 2:
        st.warning("Need at least 2 samples and 2 classes. Add more labeled cells.")
        return

    # Count, making sure every class appears (even if 0)
    counts = np.bincount(y, minlength=len(classes))

    freq_df = pd.DataFrame({"Class": list(classes), "Count": counts.astype(int)})

    st.write(f"Found {int(counts.sum())} patches across {len(classes)} classes.")
    st.table(freq_df)

    # Split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_split, stratify=y, random_state=42
    )

    # Generators
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
    go = st.button("Start training", use_container_width=True, type="primary")
    if not go:
        return

    es = EarlyStopping(
        monitor="val_loss", patience=15, restore_best_weights=True, verbose=1
    )

    with st.spinner("Training DenseNet…"):
        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=int(epochs),
            callbacks=[es],
            class_weight=class_weights_dict,
            verbose=0,
        )

    # Evaluate on validation set
    # Collect all val batches
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

    _plot_densenet_losses(
        history.history.get("loss", []),
        history.history.get("val_loss", []),
        metrics=metrics_dict,
    )

    cm = confusion_matrix(yv, y_pred, labels=np.arange(len(classes)))
    fig = _plot_confusion_matrix(cm, classes, normalize=False)
    st.pyplot(fig, use_container_width=True)

    # Keep model in session (per your requirement)
    ss["densenet_ckpt_bytes"] = model
    ss["densenet_ckpt_name"] = "densenet_finetuned"
    # Also expose a simple predictor for the rest of the app
    ss["densenet_model"] = model


# panels/train_cellpose_panel.py


def render_cellpose_train_panel(key_ns="train_cellpose"):
    st.header("Fine-tune Cellpose on your labeled data")

    if not ordered_keys():
        st.info("Upload data and label masks first.")
        return

    with st.expander("Training options", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        base_model = c1.selectbox(
            "Base model",
            options=["cyto2", "cyto", "nuclei", "scratch"],
            index=0,
        )
        epochs = c2.number_input(
            "Max epochs", 1, 500, 100, step=5, key="max_epcoh_cellpose"
        )
        lr = c3.number_input(
            "Learning rate", min_value=1e-6, max_value=1e-2, value=5e-5, format="%.5f"
        )
        wd = c4.number_input(
            "Weight decay", min_value=0.0, max_value=1.0, value=0.1, step=0.05
        )

        nimg = st.slider("Images per epoch", 1, 128, 32, 1)

    go = st.button("Start fine-tuning", use_container_width=True, type="primary")
    if not go:
        return

    recs = {k: st.session_state["images"][k] for k in ordered_keys()}

    train_losses, test_losses, model_name = finetune_cellpose_from_records(
        recs,
        base_model=base_model,
        epochs=int(epochs),
        learning_rate=lr,
        weight_decay=wd,
        nimg_per_epoch=int(nimg),
    )

    st.success(f"Fine-tuning complete ✅ (model: {model_name})")

    st.session_state["train_losses"] = train_losses
    st.session_state["test_losses"] = test_losses

    _plot_losses(train_losses, test_losses)

    masks = [rec["masks"] for rec in recs.values()]
    compare_models_mean_iou_plot(
        [rec["image"] for rec in recs.values()],
        masks,
        base_model_name=base_model if base_model != "scratch" else "cyto2",
    )
