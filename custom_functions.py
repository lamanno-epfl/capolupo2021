import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import transforms
from typing import Any, Union, List, Optional

def despline() -> None:
    ax1 = plt.gca()
    # Hide the right and top spines
    ax1.spines["right"].set_visible(False)
    ax1.spines["top"].set_visible(False)
    # Only show ticks on the left and bottom spines
    ax1.yaxis.set_ticks_position("left")
    ax1.xaxis.set_ticks_position("bottom")

def percentile_nan(x, perc=99):
    return np.percentile(np.abs(x[~np.isnan(x)]), perc)

def vector_to_image(vector, foreground_ixs, shape, fill_value=0.0):
    image = np.full(shape, fill_value)
    image.flat[foreground_ixs] = vector
    return image

def stack_to_matrix(stack, foreground_ixs):
    return np.column_stack(tuple(stack[i, :, :].flat[foreground_ixs] for i in range(stack.shape[0])))

def plot_pca_weights(
    components: np.ndarray,
    feat_names: np.ndarray = None,
    n_pc: Union[int, List, np.ndarray] = 0,
    n_top: int = 20,
    gsi: Any = None,
) -> None:
    """Plots the weights correspondent to different genes in PCA

    Parameters
    ----------
    components: np.ndarray, shape=(components, features)
        The weights / loadings of PCA

    feat_names: np.ndarray = None, shape=(features,)
        The name of the features corresponding to the component matrix

    n_pc: Union[int, List, np.ndarray] = 0
        The selected, or a list of, principal components, to be plotted

    n_top: int = 20
        The number of top weights (pos and neg to show)

    gsi: Any = None
        A Grid-Spec Object in case the plots need to be embedded in a larger figure

    """
    ixes = np.argsort(components, 1)
    topp = ixes[:, -n_top:]
    topn = ixes[:, :n_top]
    if feat_names is None:
        feat_names = np.arange(components.shape[0])

    if isinstance(n_pc, (int, float)):
        i = int(n_pc)
        if gsi is None:
            gs = plt.GridSpec(1, 3)
        else:
            gs = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gsi)

        # Overall view of the weights
        ax0 = plt.subplot(gs[0])
        vals = components[i, ixes[i, :]]
        M = np.max(np.abs(vals))
        ax0.vlines(
            np.arange(components.shape[1]),
            0,
            vals,
            color=["tab:blue" if i < 0 else "tab:red" for i in components[i, ixes[i, :]]],
        )
        despline()
        plt.axhline(0, c="k", lw=0.8)
        plt.ylabel("weights", fontsize=13)
        ax0.spines["bottom"].set_visible(False)
        ax0.xaxis.set_ticks([])
        plt.ylim(-M, M)
        plt.text(0.05, 0.88, f"PCA {i+1}", fontsize=19, transform=ax0.transAxes)
        plt.text(0.4, 0.54, f"Variables", fontsize=13, transform=ax0.transAxes)

        # Negative weights
        ax1 = plt.subplot(gs[1])
        plt.vlines(
            np.arange(topn.shape[1]), 0, components[i, topn[i, :]], color="tab:blue", lw=5
        )
        ax1.spines["right"].set_visible(False)
        ax1.spines["bottom"].set_visible(False)
        ax1.spines["top"].set_visible(False)
        ax1.xaxis.set_ticks([])
        plt.axhline(0, xmax=1.07, c="k", lw=0.75, clip_on=False)
        plt.ylim(-M, M)
        trans = transforms.blended_transform_factory(ax1.transData, ax1.transAxes)
        for j in range(topn.shape[1]):
            plt.text(j, 0.52, feat_names[topn[i, j]], transform=trans, rotation=65)

        # Positive weights
        ax2 = plt.subplot(gs[2], sharey=ax1)
        ax2.vlines(
            np.arange(topp.shape[1]), 0, components[i, topp[i, :]], color="tab:red", lw=5,
        )
        ax2.spines["left"].set_visible(False)
        ax2.spines["bottom"].set_visible(False)
        ax2.spines["top"].set_visible(False)
        ax2.tick_params(labelleft="off")
        ax2.xaxis.set_ticks([])
        ax2.spines["right"].set_visible(False)
        plt.tick_params(axis="y", labelleft=False, left=False)
        plt.axhline(0, xmin=-0.07, c="k", lw=0.75, clip_on=False)
        trans = transforms.blended_transform_factory(ax2.transData, ax2.transAxes)
        for j in range(topp.shape[1]):
            plt.text(
                j,
                0.48,
                feat_names[topp[i, j]],
                transform=trans,
                rotation=65,
                va="top",
                ha="right",
                rotation_mode="anchor",
            )

        # Axes break
        d = 0.015
        ax1.plot(
            (1.071 - d, 1.071 + d),
            (0.5 - 3 * d, 0.5 + 3 * d),
            lw=1,
            transform=ax1.transAxes,
            color="k",
            clip_on=False,
        )
        ax2.plot(
            (-0.071 - d, -0.071 + d),
            (0.5 - 3 * d, 0.5 + 3 * d),
            lw=1,
            transform=ax2.transAxes,
            color="k",
            clip_on=False,
        )
    else:
        gs = plt.GridSpec(len(n_pc), 1)
        for i in range(len(n_pc)):
            plot_pca_weights(components, feat_names, int(n_pc[i]), n_top, gs[i])

