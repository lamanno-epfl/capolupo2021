import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import transforms
from sklearn.svm import SVR
from scipy.spatial.distance import cdist
from typing import Any, Union, List, Optional, Tuple


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

def softmax(x, alpha=10):
    y = np.exp(alpha*np.array(x))
    return y / y.sum(1)[:, None]

def map_on_embedding(A, B, embedding, k_neighbours=20):
    R = cdist(A, B, metric="euclidean")
    ixR = np.argsort(R, 1)[:, :k_neighbours]
    mappings = np.median(embedding[ixR, :], 1)
    return mappings, ixR, R

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

def ixs_thatsort_a2b(a: np.ndarray, b: np.ndarray, check_content: bool = True) -> np.ndarray:
    "This is super duper magic sauce to make the order of one list to be like another"
    if check_content:
        assert len(np.intersect1d(a, b)) == len(a), f"The two arrays are not matching"
    return np.argsort(a)[np.argsort(np.argsort(b))]


def scatter_viz(x: np.ndarray, y: np.ndarray, *args: Any, **kwargs: Any) -> Any:
    """A wrapper of scatter plot that guarantees that every point is visible in a very crowded scatterplot

    Args
    ----
    x: np.ndarray
        x axis coordinates
    y: np.ndarray
        y axis coordinates
    args and kwargs:
        positional and keyword arguments as in matplotplib.pyplot.scatter

    Returns
    -------
    Plots the graph and returns the axes object
    """
    ix_x_sort = np.argsort(x, kind="mergesort")
    ix_yx_sort = np.argsort(y[ix_x_sort], kind="mergesort")
    args_new = []
    kwargs_new = {}
    for arg in args:
        if type(arg) is np.ndarray:
            args_new.append(arg[ix_x_sort][ix_yx_sort])
        else:
            args_new.append(arg)
    for karg, varg in kwargs.items():
        if type(varg) is np.ndarray:
            kwargs_new[karg] = varg[ix_x_sort][ix_yx_sort]
        else:
            kwargs_new[karg] = varg
    ax = plt.scatter(
        x[ix_x_sort][ix_yx_sort], y[ix_x_sort][ix_yx_sort], *args_new, **kwargs_new
    )
    return ax



def udistr_sample(
    x: np.ndarray,
    N: int = None,
    replace: bool = False,
    n_bins: Union[int, np.ndarray] = 15,
    random_seed: int = 19900715,
) -> np.ndarray:
    """Draw a random sample from a dataset so that its distribution is close to uniform.
       This is useful if one wants to fit a model to a dataset and the independent variable distribution is skewed

        Arguments
        ---------
        x: np.ndarray, shape=()
            The array of values

        N: int, default=None
            The number of points to sample is total.
            If None it will be determined by an heuristic that is:
            Set N so that on average for each bin one will sample as as the second least abundant bean
        
        replace: bool, default=False
            Whether to set with replacement.
            For most application `replace=False` is recommended
        
        n_bins: int or sequence, default=15
            The number of bins at which estimate the frequency of the distribution of v.
            Alternativelly a sequence of bins boundary as normally passed to np.histogram

        Returns
        -------
        ixes: np.ndarray size=(N,)
            The samples to select

    """
    if isinstance(n_bins, int):
        counts, bins = np.histogram(x, bins=n_bins)
    else:
        counts, bins = np.histogram(x, bins=n_bins)
        n_bins = len(n_bins)
    freq = counts / counts.sum()
    freq = freq[np.digitize(x, bins[1:], right=True)]
    # We want to sample with probability inversely proportional to the frequency
    freq = 1 / freq
    # normalize to 1
    freq = freq / freq.sum()
    if N is None:
        # set N so that on average for each bin one will sample as as the second least abundant bin
        N = np.sort(counts[counts > 0])[1] * n_bins
    np.random.seed(random_seed)
    ixes = np.random.choice(len(x), N, replace=replace, p=freq)
    return ixes

def filter_cv_vs_mean(
    S: np.ndarray,
    N: int,
    balanced: bool = True,
    svr_gamma: float = None,
    balanced_N: int = None,
    balanced_bins: Union[np.ndarray, int] = 7,
    plot: bool = True,
    min_nonzero_cells: int = 2,
    max_feat_avg: float = np.inf,
    min_feat_avg: float = -np.inf,
    random_seed: int = 19900715,
) -> Tuple[np.ndarray, np.ndarray]:
    """Rank genes on the basis of a CV vs mean fit, it uses a nonparametric fit (Support Vector Regression)

        Arguments
        ---------
        S: np.ndarray, shape=(features, samples)
            The data matrix
        N: int
            the number to select
        balanced: bool, default=True
            Whether to fit the cv(mean) function giving equal weights to features with averages of different scales
            even if the distribution of mean(S, 1) is not uniform
        svr_gamma: float
            the gamma hyper-parameter of the SVR
        balanced_N: int, default=7
            Optionally specify the number of variables resampled to learn the cv(mean) function when balanced == True
        balanced_bins: int or np.ndarray, default=15
            The number of bins used to balance the distribution of the mean(S, 1)
        min_nonzero_cells: int, (default=2)
            minimum number of cells that express that gene for it to be considered in the fit
        min_feat_avg: int, (default=-np.inf)
            The minimum average accepted before discarding from the the gene as not expressed
        max_feat_avg: float, (default=np.inf)
            The maximum average accepted before discarding from the the gene as house-keeping/outlier
        plot: bool, default=False
            whether to show a plot

        Returns
        -------
        cv_mean_selected: np.ndarray bool
            on the basis of the N parameter

        cv_mean_score: np.ndarray
            How much the observed CV is higher than the one predicted by a noise model fit to the data

        Note: genes excluded from the fit will have in the output the same score as the lowest scoring gene in the dataset.
    """
    muS = S.mean(1)
    detected_bool = (
        ((S > 0).sum(1) > min_nonzero_cells) & (muS < max_feat_avg) & (muS > min_feat_avg)
    )

    Sf = S[detected_bool, :]
    mu = Sf.mean(1)
    sigma = Sf.std(1, ddof=1)

    # cv = sigma / mu
    log_m = np.log2(mu)
    log_cv = np.log2(sigma) - log_m  # np.log2(cv)

    if balanced:
        ixes = udistr_sample(log_m, N=balanced_N, n_bins=balanced_bins, random_seed=random_seed)
        if svr_gamma is None:
            svr_gamma = 150.0 / len(ixes)
        svr = SVR(gamma=svr_gamma)
        svr.fit(log_m[ixes][:, None], log_cv[ixes])
        fitted_fun = svr.predict
    else:
        if svr_gamma is None:
            svr_gamma = 150.0 / len(mu)
        svr = SVR(gamma=svr_gamma)
        svr.fit(log_m[:, None], log_cv)
        fitted_fun = svr.predict
    ff = fitted_fun(log_m[:, None])
    score = log_cv - ff

    xnew = np.linspace(np.min(log_m), np.max(log_m))
    ynew = svr.predict(xnew[:, None])

    nth_score = np.sort(score)[::-1][N]

    if plot:
        plt.scatter(
            log_m[score > nth_score], log_cv[score > nth_score], s=3, alpha=0.4, c="tab:red"
        )
        plt.scatter(
            log_m[score <= nth_score], log_cv[score <= nth_score], s=3, alpha=0.4, c="tab:blue"
        )
        mu_linspace = np.linspace(np.min(log_m), np.max(log_m))
        plt.plot(mu_linspace, fitted_fun(mu_linspace[:, None]), c="k")
        plt.xlabel("log2 mean data")
        plt.ylabel("log2 CV data")

    cv_mean_score = np.zeros(detected_bool.shape)
    cv_mean_score[~detected_bool] = np.min(score) - 1e-16
    cv_mean_score[detected_bool] = score
    cv_mean_selected = cv_mean_score >= nth_score
    return cv_mean_selected, cv_mean_score