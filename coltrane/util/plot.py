import numpy as np
# flake8: noqa
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import plotly.offline as pyo
from typing import Iterator, Dict
import itertools
import plotly.express as px



class Plot:

    def __init__(
        self,
        image_width: int = 1200,
        image_height: int = 900,
        debug_mode: bool = False
    ) -> None:

        self.IMAGE_WIDTH = image_width
        self.IMAGE_HEIGHT = image_height
        self.DEBUG_MODE = debug_mode

        if not self.DEBUG_MODE:
            pyo.init_notebook_mode(connected=False)
            pio.renderers.default = 'notebook'

    def class_balance(self, balance: Dict[str, int], filename='class_balance'):
        """
        Plots and dumps labels distribution.

        Parameters
        ----------
        labels : Iterator[str]
            Ground truth label for each record.
        plot_name : string
            Dumped file name.
        """

        # labels = sorted(labels)

        labels = list(balance.keys())
        counts = list(balance.values())

        fig = go.Figure()

        bars = go.Bar(
            x=labels,
            y=counts,
            text=counts,
            textposition='auto'
        )

        fig.add_trace(bars)

        pyo.iplot(
            fig,
            filename=filename,
            image_width=self.IMAGE_WIDTH,
            image_height=self.IMAGE_HEIGHT
        )

    def features_distribution(
        self,
        records: np.ndarray,
        labels: np.ndarray,
        filename='features_distribution'
    ):
        """
        Plots features distribution.
        This method uses PCA algorithm to decompose features space.
        Each label comes with it's unique color.

        Parameters
        ----------
        records : nd.array
            Records from your data set.
        labels : nd.array
            Respective labels for each of the records.
        """

        if records.shape[1] > 2:
            decomposer = PCA(n_components=2)
            records = decomposer.fit(records).transform(records)

        labels = np.reshape(labels, (len(labels), 1))

        data_frame = np.concatenate([records, labels], axis=1)
        data_frame = pd.DataFrame(data_frame, columns=['x', 'y', 'label'])

        fig = px.scatter(data_frame, x='x', y='y', color='label')

        pyo.iplot(
            fig,
            filename=filename,
            image_width=self.IMAGE_WIDTH,
            image_height=self.IMAGE_HEIGHT
        )


    def heatmap(
        self,
        data: pd.DataFrame,
        plot_name: str,
        ylabel='',
        xlabel='',
        filename=''
    ):
        fig = go.Figure()

        columns = list(data.columns)

        heatmap = go.Heatmap(
            z=data.to_numpy(),
            y=columns,
            x=columns,
            colorscale='Viridis'
        )

        fig.add_trace(heatmap)

        fig.update_layout(
            title=plot_name
        )

        pyo.iplot(
            fig,
            filename=filename,
            image_width=self.IMAGE_WIDTH,
            image_height=self.IMAGE_HEIGHT
        )

    def metrics(self, metrics: dict, filename: str = ''):
        """
        Plots and dumps aggregated metrics.

        Parameters
        ----------
        metrics : dict
            Regrouped metrics dictionary, by label, by measure.
        """

        normalized = {}
        other = {}

        for metric, values in metrics.items():
            if np.max(values) <= 1.0:
                normalized[metric] = values
            else:
                other[metric] = values

        if normalized:
            self.boxplot(normalized)

        if other:
            self.boxplot(other)

    def boxplot(self, data: dict, filename: str = ''):

        fig = go.Figure()

        for metric, values in data.items():

            fig.add_trace(
                go.Box(
                    y=values,
                    name=metric
                )
            )

        pyo.iplot(
            fig,
            filename=filename,
            image_width=self.IMAGE_WIDTH,
            image_height=self.IMAGE_HEIGHT
        )
