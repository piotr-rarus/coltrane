from austen import Logger
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA


sns.set()


def __disposable_plot(func):
    def wrapper(*args, **kwargs):
        plt.figure(clear=True)
        func(*args, **kwargs)
        plt.close('all')

    return wrapper


@__disposable_plot
def labels_distribution(labels, logger: Logger, plot_name: str):
    """
    Plots and dumps labels distribution.
    Plot will be dumped inside logger's scope.

    Parameters
    ----------
    labels : nd.array[N, 1]
        Ground truth label for each record.
    logger : Logger
        Logger instance used to dump a plot.
    plot_name : string
        Dumped file name.
    """

    labels = sorted(labels)

    figure = sns.countplot(x=labels).get_figure()
    plt.title(plot_name)

    logger.save_fig(figure, plot_name)


@__disposable_plot
def distribution(values, logger: Logger, plot_name):

    figure = sns.distplot(values).get_figure()
    plt.title(plot_name)

    logger.save_fig(figure, plot_name)


@__disposable_plot
def features_distribution(records, labels, logger: Logger, plot_name):
    """
    Plots and dumps features distribution.
    This method uses PCA algorithm to decompose features space.
    Each label comes with it's unique color.
    Plot will be dumped inside logger's scope.

    Parameters
    ----------
    records : nd.array
        Records from your data set.
    labels : nd.array
        Respective labels for each of the records.
    logger : Logger
        Logger instance used to dump a plot.
    plot_name : string
        Dumped file name.
    """

    if records.shape[1] > 2:
        decomposer = PCA(n_components=2)
        records = decomposer.fit(records).transform(records)

    # TODO: this works only for single-label
    labels = np.reshape(labels, (len(labels), 1))

    data_set = np.concatenate([records, labels], axis=1)
    # data_set = np.hstack([records, labels])
    data_set = pd.DataFrame(data_set, columns=['X', 'Y', 'Class'])

    markers = (
        'o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P',
        'X'
    )

    markers *= 10
    label_count = len(np.unique(labels))
    markers = markers[:label_count]

    figure = sns.scatterplot(
        x=data_set.X,
        y=data_set.Y,
        hue=data_set.Class,
        style=data_set.Class,
        markers=markers
    ).get_figure()

    plt.title('Features distribution - PCA')

    logger.save_fig(figure, plot_name)


def metrics(metrics, logger: Logger, plot_name=None):
    """
    Plots and dumps aggregated metrics.

    Parameters
    ----------
    metrics : dict
        Regrouped metrics dictionary, by label, by measure.
    logger : Logger
        Logger instance used to dump a plot.
    """

    normalized = pd.DataFrame()
    other = pd.DataFrame()

    for metric, values in metrics.items():
        if np.max(values) <= 1.0:
            normalized[metric] = values
        else:
            other[metric] = values

    if not plot_name:
        plot_name = 'metrics'

    if not normalized.empty:
        boxenplot(normalized, logger, plot_name + '-normalized')

    if not other.empty:
        boxenplot(other, logger, plot_name + '-other')


def confusion_matrix(
    confusion_matrix,
    classes,
    logger: Logger
):
    """
    Plots and dumps confusion matrix. I know that side effects are bad,
    but that's just how matplotlib works internally.
    I need to handle plot disposal here, hence the need to dump plot inside
    this function.
    Plot will be dumped inside logger's scope.

    Parameters
    ----------
    confusion_matrix : nd.array
    classes : nd.array[N_classes]
        List of your class labels.
        It'll be used to fill indices and columns on plot.
    logger : Logger
        Logger instance used to dump a plot.
    """

    confusion_matrix = pd.DataFrame(
        confusion_matrix,
        index=classes,
        columns=classes
    )

    heatmap(
        confusion_matrix,
        logger,
        plot_name='confusion-matrix',
        ylabel='Ground truth',
        xlabel='Predicted'
    )


@__disposable_plot
def boxenplot(data: pd.DataFrame, logger: Logger, plot_name: str):

    ax = sns.boxenplot(data=data)

    # ax.set_xticklabels(
    #     ax.get_xticklabels(),
    #     rotation=40,
    #     ha="right",
    #     fontsize=7
    # )

    # plt.tight_layout()
    # ax.figure.autofmt_xdate()

    plt.title(plot_name)
    logger.save_fig(ax.figure, plot_name)


@__disposable_plot
def heatmap(
    data: pd.DataFrame,
    logger: Logger,
    plot_name: str,
    ylabel='',
    xlabel=''
):

    ax = sns.heatmap(data, linewidths=0.5)

    plt.title(plot_name)

    if ylabel:
        plt.ylabel(ylabel)

    if xlabel:
        plt.xlabel(xlabel)

    logger.save_fig(ax.get_figure(), plot_name, dpi=300)
