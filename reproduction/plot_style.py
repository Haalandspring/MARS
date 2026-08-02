"""Shared plotting style for publication figures.

The palette is inspired by the Nature Publishing Group style used by
ggsci, with color-blind-safe categorical colors and larger typography for
figures that will be reduced in LaTeX.
"""

from __future__ import annotations

from pathlib import Path


NPG_BLUE = "#0072B2"
NPG_CYAN = "#4DBBD5"
NPG_GREEN = "#009E73"
NPG_RED = "#D55E00"
NPG_ORANGE = "#E69F00"
NPG_PURPLE = "#8491B4"
NPG_BROWN = "#7E6148"
NPG_TAN = "#B09C85"
NPG_GREY = "#6F6F6F"
NPG_DEEP_PURPLE = "#6A3D9A"

MODEL_COLORS = {
    "rfi_shape_unet": NPG_BLUE,
    "filtool": NPG_RED,
    "rfdl": NPG_GREEN,
    "lambda0": NPG_TAN,
    "lambda1": NPG_BLUE,
    "full": NPG_BLUE,
    "no_vertical": NPG_CYAN,
    "no_horizontal": NPG_ORANGE,
    "no_decoder": NPG_DEEP_PURPLE,
}

METRIC_COLORS = {
    "precision": NPG_BLUE,
    "recall": NPG_GREEN,
    "f1": NPG_RED,
}


def apply_publication_style() -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 19,
            "axes.titlesize": 22,
            "axes.labelsize": 20,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "legend.fontsize": 18,
            "figure.titlesize": 27,
            "axes.linewidth": 1.2,
            "xtick.major.width": 1.1,
            "ytick.major.width": 1.1,
            "xtick.major.size": 5.5,
            "ytick.major.size": 5.5,
            "grid.linewidth": 0.8,
            "lines.linewidth": 2.4,
            "patch.linewidth": 0.9,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def style_axes(ax, *, grid_axis: str = "y", despine: bool = True) -> None:
    ax.grid(axis=grid_axis, alpha=0.24)
    if despine:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)


def save_publication_figure(fig, path: str | Path, *, dpi: int = 300) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
