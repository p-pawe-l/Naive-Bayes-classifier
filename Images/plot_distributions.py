import seaborn as sns 
import matplotlib.pyplot as plt 
import polars as pl
import typing as typ
import pandas as pd

class ClassDistributionPotter:
        def __init__(self, data: typ.Union[pl.DataFrame, pl.Series], storage_dir: typ.Optional[str]) -> None:
                self._data: typ.Union[pl.DataFrame, pl.Series] = data
                self._pandas_data: typ.Union[pd.DataFrame, pd.Series] = data.to_pandas()
                self._storage: typ.Optional[str] = storage_dir
                
        def draw_count_plot(self, saving_plot: bool = False) -> None:
                sns.set_style("whitegrid")
                sns.set_context("notebook", font_scale=1.1)
                fig, ax = plt.subplots(figsize=(12, 7), dpi=100)

                if isinstance(self._pandas_data, pd.Series):
                        n_colors = self._pandas_data.nunique()
                        x_col = self._pandas_data
                else:
                        n_colors = self._pandas_data[self._pandas_data.columns[0]].nunique()
                        x_col = self._pandas_data.columns[0]

                palette = sns.color_palette("husl", n_colors=n_colors)
                sns.countplot(
                        data=self._pandas_data if isinstance(self._pandas_data, pd.DataFrame) else None,
                        x=x_col,
                        palette=palette,
                        ax=ax,
                        edgecolor='white',
                        linewidth=2
                )

                for container in ax.containers:
                    ax.bar_label(container, fmt='%d', padding=5, fontsize=11, fontweight='bold')

                ax.set_title(
                        "Class Distribution Analysis",
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
                        "Number of Occurrences",
                        fontsize=14,
                        fontweight='semibold',
                        color='#34495E',
                        labelpad=10
                )

                ax.grid(True, axis='y', alpha=0.3, linestyle='--', linewidth=0.7)
                ax.set_axisbelow(True)
                sns.despine(left=False, bottom=False, top=True, right=True)
                ax.tick_params(axis='both', labelsize=11, colors='#34495E')
                ax.set_facecolor('#FAFAFA')
                fig.patch.set_facecolor('white')

                plt.tight_layout()

                if saving_plot:
                        plt.savefig(
                                f'{self._storage}/CountPlot.png',
                                dpi=300,
                                bbox_inches='tight',
                                facecolor='white',
                                edgecolor='none'
                        )

                plt.show()
        
        def draw_circle_plot(self, saving_plot: bool = False) -> None:
                sns.set_style("whitegrid")
                sns.set_context("notebook", font_scale=1.1)
                fig, ax = plt.subplots(figsize=(12, 7), dpi=100)

                if isinstance(self._pandas_data, pd.Series):
                        n_colors = self._pandas_data.nunique()
                        value_counts = self._pandas_data.value_counts()
                        labels = value_counts.index.tolist()
                        sizes = value_counts.values.tolist()
                else:
                        n_colors = self._pandas_data[self._pandas_data.columns[0]].nunique()
                        value_counts = self._pandas_data[self._pandas_data.columns[0]].value_counts()
                        labels = value_counts.index.tolist()
                        sizes = value_counts.values.tolist()

                palette = sns.color_palette("husl", n_colors=n_colors)

                _, _, autotexts = ax.pie(
                        sizes,
                        labels=labels,
                        colors=palette,
                        autopct='%1.1f%%',
                        startangle=90,
                        pctdistance=0.85,
                        explode=[0.05] * len(sizes),
                        wedgeprops=dict(width=0.7, edgecolor='white', linewidth=2),
                        textprops=dict(color='#34495E', fontsize=11, fontweight='semibold')
                )

                for autotext in autotexts:
                        autotext.set_color('white')
                        autotext.set_fontsize(12)
                        autotext.set_fontweight('bold')

                ax.set_title(
                        "Class Distribution Proportion",
                        fontsize=18,
                        fontweight='bold',
                        pad=20,
                        color='#2C3E50'
                )

                centre_circle = plt.Circle((0, 0), 0.70, fc='white')
                ax.add_artist(centre_circle)

                total_count = sum(sizes)
                ax.text(
                        0, 0,
                        f'Total\n{total_count}',
                        ha='center',
                        va='center',
                        fontsize=16,
                        fontweight='bold',
                        color='#2C3E50'
                )

                ax.axis('equal')
                fig.patch.set_facecolor('white')

                plt.tight_layout()

                if saving_plot:
                        plt.savefig(
                                f'{self._storage}/CirclePlot.png',
                                dpi=300,
                                bbox_inches='tight',
                                facecolor='white',
                                edgecolor='none'
                        )

                plt.show()
