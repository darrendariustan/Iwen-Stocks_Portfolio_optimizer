"""
The ``plotting`` module houses all the functions to generate various plots.

Currently implemented:

  - ``plot_covariance`` - plot a correlation matrix
  - ``plot_dendrogram`` - plot the hierarchical clusters in a portfolio
  - ``plot_efficient_frontier`` – plot the efficient frontier from an EfficientFrontier or CLA object
  - ``plot_weights`` - bar chart of weights
"""

import warnings
import copy

import numpy as np
import scipy.cluster.hierarchy as sch

from . import CLA, EfficientFrontier, exceptions, risk_models

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    # Register seaborn styles
    sns.set_theme()
    try:
        plt.style.use("seaborn-deep")
    except OSError:
        plt.style.use("default")
except (ModuleNotFoundError, ImportError):  # pragma: no cover
    raise ImportError("Please install matplotlib via pip or poetry")


def _get_plotly():
    """Helper function to import plotly only when needed"""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        return go, make_subplots
    except (ModuleNotFoundError, ImportError):
        raise ImportError(
            "Please install plotly via pip or poetry to use interactive plots"
        )


def _plot_io(**kwargs):
    """
    Helper method to optionally save the figure to file.

    :param filename: name of the file to save to, defaults to None (doesn't save)
    :type filename: str, optional
    :param dpi: dpi of figure to save or plot, defaults to 300
    :type dpi: int (between 50-500)
    :param showfig: whether to plt.show() the figure, defaults to False
    :type showfig: bool, optional
    """
    filename = kwargs.get("filename", None)
    showfig = kwargs.get("showfig", False)
    dpi = kwargs.get("dpi", 300)

    plt.tight_layout()
    if filename:
        plt.savefig(fname=filename, dpi=dpi)
    if showfig:  # pragma: no cover
        plt.show()


def plot_covariance(cov_matrix, plot_correlation=False, show_tickers=True, **kwargs):
    """
    Generate a basic plot of the covariance (or correlation) matrix, given a
    covariance matrix.

    :param cov_matrix: covariance matrix
    :type cov_matrix: pd.DataFrame or np.ndarray
    :param plot_correlation: whether to plot the correlation matrix instead, defaults to False.
    :type plot_correlation: bool, optional
    :param show_tickers: whether to use tickers as labels (not recommended for large portfolios),
                        defaults to True
    :type show_tickers: bool, optional

    :return: matplotlib axis
    :rtype: matplotlib.axes object
    """
    if plot_correlation:
        matrix = risk_models.cov_to_corr(cov_matrix)
    else:
        matrix = cov_matrix
    fig, ax = plt.subplots()

    cax = ax.imshow(matrix)
    fig.colorbar(cax)

    if show_tickers:
        ax.set_xticks(np.arange(0, matrix.shape[0], 1))
        ax.set_xticklabels(matrix.index)
        ax.set_yticks(np.arange(0, matrix.shape[0], 1))
        ax.set_yticklabels(matrix.index)
        plt.xticks(rotation=90)

    _plot_io(**kwargs)

    return ax


def plot_dendrogram(hrp, ax=None, show_tickers=True, **kwargs):
    """
    Plot the clusters in the form of a dendrogram.

    :param hrp: HRPpt object that has already been optimized.
    :type hrp: object
    :param show_tickers: whether to use tickers as labels (not recommended for large portfolios),
                        defaults to True
    :type show_tickers: bool, optional
    :param filename: name of the file to save to, defaults to None (doesn't save)
    :type filename: str, optional
    :param showfig: whether to plt.show() the figure, defaults to False
    :type showfig: bool, optional
    :return: matplotlib axis
    :rtype: matplotlib.axes object
    """
    if ax is None:
        fig, ax = plt.subplots()

    # Compute linkage matrix
    distance = hrp.discretize_distance(hrp.cov)
    linkage = sch.linkage(distance, "single")

    # Get ticker labels if required
    labels = hrp.tickers if show_tickers else None

    # Plot the dendrogram
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dn = sch.dendrogram(linkage, orientation="top", ax=ax, labels=labels, leaf_rotation=90)

    if show_tickers:
        ax.set_xticklabels(labels, rotation=90)

    _plot_io(**kwargs)

    return ax


def _plot_cla(cla, points, ax, show_assets, show_tickers, interactive):
    """
    Helper function to plot the efficient frontier from a CLA object
    """
    if interactive:
        go, make_subplots = _get_plotly()
        fig = go.Figure()
    else:
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = plt.gcf()

    # Extract risk and returns from the CLA object
    risks = [0] * points
    rets = [0] * points
    weights = [None] * points

    # Evenly sample the frontier by looping through the weights
    # and finding the returns/risks at those weights.
    i = 0
    retmin, retmax = cla.bounds()
    step = (retmax - retmin) / points
    for target_return in np.arange(retmin, retmax, step):
        weights[i] = cla.efficient_return(target_return)
        portfolio_risk = np.sqrt(np.dot(weights[i], np.dot(cla.cov_matrix, weights[i].T)))
        risks[i] = portfolio_risk
        rets[i] = np.sum(np.array(weights[i]) * np.array(cla.expected_returns), axis=0)
        i += 1

    if interactive:
        data = {"risks": risks[:i], "rets": rets[:i]}
        fig.add_trace(
            go.Scatter(
                x=data["risks"], y=data["rets"], mode="lines", opacity=1, name="Efficient Frontier"
            )
        )
    else:
        ax.plot(risks[:i], rets[:i], label="Efficient frontier")
        ax.set(
            title="Efficient Frontier",
            xlabel="Volatility",
            ylabel="Return",
            xlim=[0, max(risks[:i]) * 1.2],
            ylim=[min(rets[:i]) * 0.8, max(rets[:i]) * 1.2],
        )
        fig.tight_layout()

    if show_assets and cla.tickers is not None:
        asset_risks = np.sqrt(np.diag(cla.cov_matrix))
        asset_rets = cla.expected_returns

        if interactive:
            text = cla.tickers
            fig.add_trace(
                go.Scatter(
                    x=asset_risks,
                    y=asset_rets,
                    mode="markers",
                    text=text,
                    name="Assets",
                    marker=dict(colorscale="viridis", size=10),
                )
            )
        else:
            ax.scatter(asset_risks, asset_rets, label="Assets", alpha=0.8)
            if show_tickers:
                for ticker, x, y in zip(cla.tickers, asset_risks, asset_rets):
                    ax.annotate(ticker, (x, y), xytext=(1, 1), textcoords="offset points")
            ax.legend()

    if interactive:
        fig.update_layout(
            title="Efficient Frontier",
            xaxis=dict(title="Volatility", range=[0, max(risks[:i]) * 1.2]),
            yaxis=dict(title="Return", range=[min(rets[:i]) * 0.8, max(rets[:i]) * 1.2]),
            colorway=px.colors.qualitative.G10,
            width=900,
            height=600,
        )
        return fig
    else:
        return ax


def _ef_default_returns_range(ef, points):
    """
    Helper function to generate a range of returns from the GMV returns to
    the maximum (constrained) returns
    """
    # Get minimum volatility portfolio
    ef_gmv = copy.deepcopy(ef)
    ef_gmv.min_volatility()
    min_ret = ef_gmv.portfolio_performance()[0]
    
    # Try to get max return portfolio safely
    ef_max = copy.deepcopy(ef)
    try:
        # Try max sharpe first (safer than custom_objective)
        ef_max.max_sharpe()
        max_ret = ef_max.portfolio_performance()[0]
    except exceptions.OptimizationError:
        # Fallback: try a higher return than min_vol
        try:
            max_ret = min_ret * 1.5  # Estimate 50% higher return
            ef_max.efficient_return(max_ret)
            max_ret = ef_max.portfolio_performance()[0]
        except exceptions.OptimizationError:
            # Ultra-safe fallback
            max_ret = min_ret * 1.1  # Just 10% higher than min
    
    # Ensure we have a reasonable range
    if max_ret <= min_ret:
        max_ret = min_ret * 1.1
        
    return np.linspace(min_ret, max_ret, points)


def _plot_ef(ef, ef_param, ef_param_range, ax, show_assets, show_tickers, interactive):
    """
    Helper function to plot the efficient frontier from an EfficientFrontier object
    """
    if interactive:
        go, make_subplots = _get_plotly()
        fig = go.Figure()
    else:
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = plt.gcf()

    if ef_param in {"utility", "return"}:
        # Calculate the optimal weights for a range of desired returns/risk aversions
        if ef_param == "utility" and ef_param_range is None:
            ef_param_range = np.linspace(1, 10, 20)
        elif ef_param == "return" and ef_param_range is None:
            ef_param_range = _ef_default_returns_range(ef, 20)

        # Plot the efficient frontier
        rets = []
        risks = []

        if ef_param == "utility":
            for delta in ef_param_range:
                ef_i = copy.deepcopy(ef)
                try:
                    ef_i.max_sharpe(risk_free_rate=0, risk_aversion=delta)
                    # Use the portfolio_performance method directly from EfficientFrontier
                    ret, risk, _ = ef_i.portfolio_performance()
                    rets.append(ret)
                    risks.append(risk)
                except exceptions.OptimizationError:
                    continue
        elif ef_param == "return":
            for ret_i in ef_param_range:
                ef_i = copy.deepcopy(ef)
                try:
                    ef_i.efficient_return(ret_i)
                    # Use the portfolio_performance method directly from EfficientFrontier
                    ret, risk, _ = ef_i.portfolio_performance()
                    rets.append(ret)
                    risks.append(risk)
                except exceptions.OptimizationError:
                    continue

    elif ef_param == "risk":
        # Unlike for utility or return, we cannot leverage the copy method,
        # as both the constraints and the objective change, because
        # tight variance bounds result in failure to converge.
        # So we first create a vector of variances, and then calculate the
        # efficient returns for each variance.
        if ef_param_range is None:
            # The minimum variance
            ef_minvol = copy.deepcopy(ef)
            ef_minvol.min_volatility()
            min_var = ef_minvol.portfolio_performance()[1] ** 2

            # The variance of the tangency portfolio
            try:
                ef_sharpe = copy.deepcopy(ef)
                ef_sharpe.max_sharpe()
                max_var = ef_sharpe.portfolio_performance()[1] ** 2
            except exceptions.OptimizationError:
                max_var = min_var * 20

            variances = np.linspace(min_var, max_var, 20)
        else:
            # Convert target risk to target var
            variances = [target_risk ** 2 for target_risk in ef_param_range]

        risks = []
        rets = []
        for var in variances:
            ef_i = copy.deepcopy(ef)
            try:
                w = ef_i.efficient_risk(np.sqrt(var))
                if w is not None:
                    # Use the portfolio_performance method directly from EfficientFrontier
                    ret, risk, _ = ef_i.portfolio_performance()
                    rets.append(ret)
                    risks.append(risk)
            except exceptions.OptimizationError:
                continue

    if interactive:
        data = {"risks": risks, "rets": rets}
        fig.add_trace(
            go.Scatter(
                x=data["risks"], y=data["rets"], mode="lines", opacity=1, name="Efficient Frontier"
            )
        )
    else:
        ax.plot(risks, rets, label="Efficient frontier")
        ax.set(
            title="Efficient Frontier",
            xlabel="Volatility",
            ylabel="Return",
            xlim=[0, max(risks) * 1.1],
            ylim=[min(rets) * 0.6, max(rets) * 1.1],
        )

    if show_assets and ef.tickers is not None:
        asset_risks = np.sqrt(np.diag(ef.cov_matrix))
        asset_rets = ef.expected_returns

        if interactive:
            text = ef.tickers
            fig.add_trace(
                go.Scatter(
                    x=asset_risks,
                    y=asset_rets,
                    mode="markers",
                    text=text,
                    name="Assets",
                    marker=dict(colorscale="viridis", size=10),
                )
            )
        else:
            ax.scatter(asset_risks, asset_rets, label="Assets", alpha=0.8)
            if show_tickers:
                for ticker, x, y in zip(ef.tickers, asset_risks, asset_rets):
                    ax.annotate(ticker, (x, y), xytext=(1, 1), textcoords="offset points")
            ax.legend()

    if interactive:
        fig.update_layout(
            title="Efficient Frontier",
            xaxis=dict(title="Volatility", range=[0, max(risks) * 1.1]),
            yaxis=dict(title="Return", range=[min(rets) * 0.6, max(rets) * 1.1]),
            colorway=px.colors.qualitative.G10,
            width=900,
            height=600,
        )
        return fig
    else:
        return ax


def plot_efficient_frontier(
    opt,
    ef_param="return",
    ef_param_range=None,
    points=100,
    ax=None,
    show_assets=True,
    show_tickers=False,
    interactive=False,
    **kwargs,
):
    """
    Plot the efficient frontier based on either a CLA or EfficientFrontier object.

    :param opt: an instantiated optimizer object BEFORE optimising an objective
    :type opt: EfficientFrontier or CLA
    :param ef_param: [EfficientFrontier] whether to use a range over utility, risk, or return.
                     Defaults to "return".
    :type ef_param: str, one of {"utility", "risk", "return"}.
    :param ef_param_range: the range of parameter values for ef_param.
                           If None, automatically compute a range from min->max return.
    :type ef_param_range: np.array or list (recommended to use np.arange or np.linspace)
    :param points: number of points to plot, defaults to 100. This is overridden if
                   an `ef_param_range` is provided explicitly.
    :type points: int, optional
    :param show_assets: whether we should plot the asset risks/returns also, defaults to True
    :type show_assets: bool, optional
    :param show_tickers: whether we should annotate each asset with its ticker, defaults to False
    :type show_tickers: bool, optional
    :param interactive: Switch rendering engine between Plotly and Matplotlib
    :type show_tickers: bool, optional
    :param filename: name of the file to save to, defaults to None (doesn't save)
    :type filename: str, optional
    :param showfig: whether to plt.show() the figure, defaults to False
    :type showfig: bool, optional
    :return: matplotlib axis
    :rtype: matplotlib.axes object
    """
    if points < 10:
        warnings.warn("When plotting efficient frontier, you should use at least 10 points")

    if interactive:
        try:
            import plotly.express as px  # noqa
        except ImportError:
            raise ImportError(
                "Please install plotly via pip or poetry to use interactive plots"
            )

    if isinstance(opt, CLA):
        if ef_param != "return":
            warnings.warn(
                "When using CLA, ef_param is always return."
                + " Your input has been ignored."
            )
        return _plot_cla(
            cla=opt,
            points=points,
            ax=ax,
            show_assets=show_assets,
            show_tickers=show_tickers,
            interactive=interactive,
        )
    elif isinstance(opt, EfficientFrontier):
        if ef_param not in {"utility", "risk", "return"}:
            raise ValueError("ef_param should be one of {'utility', 'risk', 'return'}")
        return _plot_ef(
            ef=opt,
            ef_param=ef_param,
            ef_param_range=ef_param_range,
            ax=ax,
            show_assets=show_assets,
            show_tickers=show_tickers,
            interactive=interactive,
        )
    else:
        raise TypeError("opt should be an instance of EfficientFrontier or CLA")


def plot_weights(weights, ax=None, **kwargs):
    """
    Plot the portfolio weights as a horizontal bar chart

    :param weights: the weights outputted by any PyPortfolioOpt optimizer
    :type weights: {ticker: weight} dict
    :param ax: ax to plot to, optional
    :type ax: matplotlib.axes
    :return: matplotlib axis
    :rtype: matplotlib.axes
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = plt.gcf()

    desc = sorted(weights.items(), key=lambda x: x[1], reverse=True)
    labels = [i[0] for i in desc]
    vals = [i[1] for i in desc]

    y_pos = np.arange(len(labels))
    ax.barh(y_pos, vals)
    ax.set_xlabel("Weight")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()

    _plot_io(**kwargs)
    return ax
