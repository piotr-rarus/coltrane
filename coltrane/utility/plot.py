from austen import Logger
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA


sns.set()


def labels_distribution(labels, logger: Logger, plot_name):
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

    plt.figure(clear=True)

    figure = sns.countplot(x=labels).get_figure()

    plt.title('Class balance')

    logger.save_fig(figure, plot_name)
    plt.close('all')


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

    plt.figure(clear=True)

    if records.shape[1] > 2:
        decomposer = PCA(n_components=2)
        records = decomposer.fit(records).transform(records)

    # TODO: this works only for single-label
    labels = np.reshape(labels, (len(labels), 1))

    data_set = np.concatenate([records, labels], axis=1)
    # data_set = np.hstack([records, labels])
    data_set = pd.DataFrame(data_set, columns=['X', 'Y', 'Class'])

    figure = sns.scatterplot(
        x=data_set.X,
        y=data_set.Y,
        hue=data_set.Class,
        style=data_set.Class
    ).get_figure()

    plt.title('Features distribution - PCA')

    logger.save_fig(figure, plot_name)
    plt.close('all')


def stats(stats, logger: Logger):
    """
    Plots and dumps aggregated metrics.

    Parameters
    ----------
    stats : dict
        Regrouped stats dictionary, by label, by measure.
    logger : Logger
        Logger instance used to dump a plot.
    """

    for label, measures in stats.items():
        data_frame = pd.DataFrame()

        for measure, values in measures.items():
            if measure not in ['support']:
                data_frame[measure] = values

        plt.figure(clear=True)

        figure = sns.boxenplot(data=data_frame).figure

        plt.title(label)

        logger.save_fig(figure, 'stats - ' + label)
        plt.close('all')


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

    plt.figure(clear=True)

    confusion_matrix = pd.DataFrame(
        confusion_matrix,
        index=classes,
        columns=classes
    )

    figure = sns.heatmap(confusion_matrix, annot=True).get_figure()

    plt.title('Confusion matrix')
    plt.ylabel('Ground truth')
    plt.xlabel('Predicted')

    logger.save_fig(figure, 'confusion_matrix')
    plt.close('all')
