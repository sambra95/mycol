import io
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import streamlit as st
from helpers.state_ops import ordered_keys
from skimage.measure import regionprops
from zipfile import ZipFile
from pathlib import Path
from zipfile import ZIP_DEFLATED


def plot_violin(df: pd.DataFrame, value_col: str):
    sub = df.copy()
    sub["label"] = sub["mask label"].replace("No label", None).fillna("Unlabelled")
    order = sorted(sub["label"].unique(), key=lambda x: (x != "Unlabelled", str(x)))

    palette = px.colors.qualitative.Set2
    color_map = {lab: palette[i % len(palette)] for i, lab in enumerate(order)}

    show_points = bool(st.session_state.get("overlay_datapoints", False))

    fig = go.Figure()

    for lab in order:
        vals = sub.loc[sub["label"] == lab, value_col]
        x_vals = [lab] * len(vals)

        # 1) VIOLIN SHAPE (non-interactive, no points)
        fig.add_trace(
            go.Violin(
                x=x_vals,
                y=vals,
                name=str(lab),
                legendgroup=str(lab),
                box_visible=True,
                meanline_visible=True,
                line_color="black",
                fillcolor=color_map[lab],
                opacity=0.85,
                points=False,  # no points from this trace
                hoverinfo="skip",  # no hover on the shape
                showlegend=False,
            )
        )

        # 2) POINTS-ONLY VIOLIN (interactive, transparent body/outline)
        if show_points and len(vals) > 0:
            fig.add_trace(
                go.Violin(
                    x=x_vals,
                    y=vals,
                    name=f"{lab}_pts",
                    legendgroup=str(lab),
                    box_visible=False,
                    meanline_visible=False,
                    line_color="rgba(0,0,0,0)",  # hide outline
                    fillcolor="rgba(0,0,0,0)",  # no fill
                    opacity=1.0,
                    points="all",  # show points
                    pointpos=0,  # centered over violin
                    jitter=0.25,  # horizontal spread
                    marker=dict(
                        size=4,
                        opacity=0.8,
                        color="black",
                        line=dict(width=0.3, color="black"),
                    ),
                    showlegend=False,
                )
            )

    fig.update_layout(
        violinmode="overlay",
        xaxis_title="Label",
        yaxis_title=value_col.replace("_", " ").title(),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=40, r=40, t=40, b=40),
        height=500,
        showlegend=False,
    )
    fig.update_xaxes(showline=True, linecolor="black", gridcolor="rgba(0,0,0,0.1)")
    fig.update_yaxes(showline=True, linecolor="black", gridcolor="rgba(0,0,0,0.1)")

    return f"{value_col.replace(' ', '_')}", fig


def plot_bar(df: pd.DataFrame, value_col: str):
    sub = df.copy()
    sub["label"] = sub["mask label"].replace("No label", None).fillna("Unlabelled")
    order = sorted(sub["label"].unique(), key=lambda x: (x != "Unlabelled", str(x)))
    pal = sns.color_palette("Set2", n_colors=len(order))

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    sns.barplot(
        data=sub, x="label", y=value_col, order=order, palette=pal, errorbar="sd", ax=ax
    )
    for p in ax.patches:
        p.set_edgecolor("black")
        p.set_linewidth(0.8)

    # Only overlay individual points if toggle is on
    if st.session_state["overlay_datapoints"]:
        sns.stripplot(
            data=sub,
            x="label",
            y=value_col,
            order=order,
            color="k",
            alpha=0.6,
            size=3,
            jitter=0.25,
            linewidth=0.3,
            edgecolor="black",
            ax=ax,
        )

    ax.set_xlabel("Label")
    ax.set_ylabel(value_col.replace("_", " ").title())
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    sns.despine(ax=ax)

    st.pyplot(fig)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return f"bar_{value_col.replace(' ', '_')}.png", buf.getvalue()


def build_analysis_df():
    rows = []
    for k in ordered_keys():
        rec = st.session_state.images[k]
        inst = rec.get("masks")
        if not isinstance(inst, np.ndarray) or inst.ndim != 2 or not inst.any():
            continue

        labdict = rec.get("labels", {})  # dict {instance_id -> class/None}
        for prop in regionprops(inst):  # prop.label is the instance id
            iid = int(prop.label)
            cls = labdict.get(iid)
            rows.append(
                {
                    "image": rec["name"],
                    "mask #": iid,
                    "mask label": ("Unlabelled" if cls in (None, "No label") else cls),
                    "mask area": float(prop.area),
                    "mask perimeter": float(
                        prop.perimeter
                    ),  # or perimeter_crofton if you prefer
                    # add any other metrics here, using `prop`
                }
            )
    return pd.DataFrame(rows)


def build_image_summary_df():
    rows = []
    all_classes = set()

    for k in ordered_keys():
        rec = st.session_state.images[k]
        inst = rec.get("masks")
        if not isinstance(inst, np.ndarray) or inst.ndim != 2:
            continue

        ids = np.unique(inst)
        ids = ids[ids != 0]
        total = len(ids)

        labdict = rec.get("labels", {})
        counts = {}
        unlabelled = 0

        for iid in ids:
            cls = labdict.get(int(iid))
            if cls is None or cls == "No label":
                unlabelled += 1
            else:
                counts[cls] = counts.get(cls, 0) + 1
                all_classes.add(cls)

        rows.append(
            {
                "image": rec["name"],
                "total cells": total,
                "unlabelled": unlabelled,
                **counts,
            }
        )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).fillna(0)
    # make sure integer counts
    for col in df.columns:
        if col != "image":
            df[col] = df[col].astype(int)

    # ensure all class columns appear
    for cls in sorted(all_classes):
        if cls not in df.columns:
            df[cls] = 0

    return df


# --- FUNCTIONS FOR DOWNLOADING CLASS CHARACTERISTICS PLOTS


def build_cell_metrics_zip(labels_selected):
    df = build_analysis_df()
    if labels_selected:
        df = df[df["mask label"].isin(labels_selected)]
    items = []
    if not df.empty:
        items.append(("cell_analysis.csv", df.to_csv(index=False).encode("utf-8")))
    counts_df = build_image_summary_df()
    if not counts_df.empty:
        items.append(
            ("image_counts.csv", counts_df.to_csv(index=False).encode("utf-8"))
        )
    items += st.session_state.get("analysis_plots", [])
    return build_plots_zip(items) if items else b""


def build_plots_zip(plot_paths_or_bytes) -> bytes:
    """
    Accepts either list of file paths or list of (name, bytes).
    """
    if not plot_paths_or_bytes:
        return b""
    buf = io.BytesIO()
    with ZipFile(buf, mode="w", compression=ZIP_DEFLATED) as zf:
        for i, item in enumerate(plot_paths_or_bytes):
            if isinstance(item, (str, Path)) and Path(item).exists():
                p = Path(item)
                zf.writestr(p.name, p.read_bytes())
            elif isinstance(item, tuple) and len(item) == 2:
                zf.writestr(str(item[0]), item[1])
            else:
                # skip unknown
                pass
    return buf.getvalue()
