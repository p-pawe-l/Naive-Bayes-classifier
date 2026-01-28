import seaborn as sns
import matplotlib.pyplot as plt
import polars as pl
import typing as typ
import pandas as pd
import numpy as np

class MathStatisticsPlotter:
        def __init__(self, data: pl.DataFrame, storage_dir: typ.Optional[str]) -> None:
                self._data: pl.DataFrame = data
                self._pandas_data: pd.DataFrame = data.to_pandas()
                self._storage: typ.Optional[str] = storage_dir
                self._prepare_statistics()

        def _prepare_statistics(self) -> None:
                self._pandas_data['message_length'] = self._pandas_data['v2'].str.len()

                stats_data = []
                for class_name in self._pandas_data['v1'].unique():
                        class_lengths = self._pandas_data[self._pandas_data['v1'] == class_name]['message_length']

                        stats_data.append({
                                'Class': class_name,
                                'Min': class_lengths.min(),
                                'Max': class_lengths.max(),
                                'Mean': class_lengths.mean(),
                                'Median': class_lengths.median(),
                                'Variance': class_lengths.var(),
                                'Std Dev': class_lengths.std()
                        })

                self._stats_df = pd.DataFrame(stats_data)

        def draw_basic_statistics_plot(self, saving_plot: bool = False) -> None:
                sns.set_style("whitegrid")
                sns.set_context("notebook", font_scale=1.1)
                fig, ax = plt.subplots(figsize=(12, 7), dpi=100)

                stats_to_plot = ['Min', 'Max', 'Mean', 'Median']
                x = np.arange(len(self._stats_df['Class']))
                width = 0.2

                colors = sns.color_palette("husl", n_colors=len(stats_to_plot))

                for i, stat in enumerate(stats_to_plot):
                        offset = width * (i - len(stats_to_plot) / 2 + 0.5)
                        bars = ax.bar(
                                x + offset,
                                self._stats_df[stat],
                                width,
                                label=stat,
                                color=colors[i],
                                edgecolor='white',
                                linewidth=1.5
                        )

                        ax.bar_label(bars, fmt='%.1f', padding=3, fontsize=9, fontweight='bold')

                ax.set_title(
                        "Message Length Statistics by Class",
                        fontsize=18,
                        fontweight='bold',
                        pad=20,
                        color='#2C3E50'
                )
                ax.set_xlabel(
                        "Class Label",
                        fontsize=14,
                        fontweight='semibold',
                        color='#34495E',
                        labelpad=10
                )
                ax.set_ylabel(
                        "Message Length (characters)",
                        fontsize=14,
                        fontweight='semibold',
                        color='#34495E',
                        labelpad=10
                )

                ax.set_xticks(x)
                ax.set_xticklabels(self._stats_df['Class'])
                ax.legend(loc='upper left', frameon=True, shadow=True, fontsize=11)

                ax.grid(True, axis='y', alpha=0.3, linestyle='--', linewidth=0.7)
                ax.set_axisbelow(True)
                sns.despine(left=False, bottom=False, top=True, right=True)
                ax.tick_params(axis='both', labelsize=11, colors='#34495E')
                ax.set_facecolor('#FAFAFA')
                fig.patch.set_facecolor('white')

                plt.tight_layout()

                if saving_plot:
                        plt.savefig(
                                f'{self._storage}/BasicStatisticsPlot.png',
                                dpi=300,
                                bbox_inches='tight',
                                facecolor='white',
                                edgecolor='none'
                        )

                plt.show()

        def draw_variance_std_plot(self, saving_plot: bool = False) -> None:
                sns.set_style("whitegrid")
                sns.set_context("notebook", font_scale=1.1)
                fig, ax = plt.subplots(figsize=(12, 7), dpi=100)

                x = np.arange(len(self._stats_df['Class']))
                width = 0.35

                colors = sns.color_palette("husl", n_colors=2)

                bars1 = ax.bar(
                        x - width/2,
                        self._stats_df['Variance'],
                        width,
                        label='Variance',
                        color=colors[0],
                        edgecolor='white',
                        linewidth=2
                )

                bars2 = ax.bar(
                        x + width/2,
                        self._stats_df['Std Dev'],
                        width,
                        label='Standard Deviation',
                        color=colors[1],
                        edgecolor='white',
                        linewidth=2
                )

                ax.bar_label(bars1, fmt='%.2f', padding=5, fontsize=11, fontweight='bold')
                ax.bar_label(bars2, fmt='%.2f', padding=5, fontsize=11, fontweight='bold')

                ax.set_title(
                        "Variance and Standard Deviation by Class",
                        fontsize=18,
                        fontweight='bold',
                        pad=20,
                        color='#2C3E50'
                )
                ax.set_xlabel(
                        "Class Label",
                        fontsize=14,
                        fontweight='semibold',
                        color='#34495E',
                        labelpad=10
                )
                ax.set_ylabel(
                        "Value",
                        fontsize=14,
                        fontweight='semibold',
                        color='#34495E',
                        labelpad=10
                )

                ax.set_xticks(x)
                ax.set_xticklabels(self._stats_df['Class'])
                ax.legend(loc='upper left', frameon=True, shadow=True, fontsize=11)

                ax.grid(True, axis='y', alpha=0.3, linestyle='--', linewidth=0.7)
                ax.set_axisbelow(True)
                sns.despine(left=False, bottom=False, top=True, right=True)
                ax.tick_params(axis='both', labelsize=11, colors='#34495E')
                ax.set_facecolor('#FAFAFA')
                fig.patch.set_facecolor('white')

                plt.tight_layout()

                if saving_plot:
                        plt.savefig(
                                f'{self._storage}/VarianceStdPlot.png',
                                dpi=300,
                                bbox_inches='tight',
                                facecolor='white',
                                edgecolor='none'
                        )

                plt.show()
