import seaborn as sns
import matplotlib.pyplot as plt
import polars as pl
import typing as typ
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import classifier as cl
import Token.tokenization as tok


class ExperimentPlotter:
        def __init__(self, data: pl.DataFrame, storage_dir: typ.Optional[str]) -> None:
                self._data: pl.DataFrame = data
                self._pandas_data: pd.DataFrame = data.to_pandas()
                self._storage: typ.Optional[str] = storage_dir
                self._experiment_results: typ.Dict[str, typ.Any] = {}

        def run_tokenizer_comparison(self, test_size: float = 0.2) -> None:
                tokenizers = {
                        'Unigram': tok.UnigramTokenization,
                        'Bigram': tok.BigramTokenization
                }

                results = {
                        'Tokenizer': [],
                        'Accuracy': [],
                        'Recall': [],
                        'Precision': [],
                        'F1 Score': [],
                        'Training Time': []
                }

                X = self._data['v2_cleaned_tokenized'].to_list()
                y = self._data['v1'].to_list()
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

                for name, tokenizer_class in tokenizers.items():
                        classifier = cl.ClassifierNB(tokenizer=tokenizer_class)

                        start_time = time.time()
                        classifier.fit(X_train, y_train)
                        training_time = time.time() - start_time

                        predictions = []
                        for sample in X_test:
                                pred_dict = classifier.predict(sample)
                                predictions.append(max(pred_dict, key=pred_dict.get))

                        accuracy = accuracy_score(y_test, predictions)
                        recall = recall_score(y_test, predictions, average='weighted', zero_division=0)
                        precision = precision_score(y_test, predictions, average='weighted', zero_division=0)
                        f1 = f1_score(y_test, predictions, average='weighted', zero_division=0)

                        results['Tokenizer'].append(name)
                        results['Accuracy'].append(accuracy)
                        results['Recall'].append(recall)
                        results['Precision'].append(precision)
                        results['F1 Score'].append(f1)
                        results['Training Time'].append(training_time)

                self._experiment_results['tokenizer_comparison'] = pd.DataFrame(results)

        def run_test_size_comparison(self) -> None:
                test_sizes = [0.1, 0.2, 0.3, 0.4]

                results = {
                        'Test Size': [],
                        'Accuracy': [],
                        'Recall': [],
                        'Training Time': [],
                        'Training Samples': [],
                        'Test Samples': []
                }

                X = self._data['v2_cleaned_tokenized'].to_list()
                y = self._data['v1'].to_list()

                for test_size in test_sizes:
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

                        classifier = cl.ClassifierNB(tokenizer=tok.UnigramTokenization)

                        start_time = time.time()
                        classifier.fit(X_train, y_train)
                        training_time = time.time() - start_time

                        predictions = []
                        for sample in X_test:
                                pred_dict = classifier.predict(sample)
                                predictions.append(max(pred_dict, key=pred_dict.get))

                        accuracy = accuracy_score(y_test, predictions)
                        recall = recall_score(y_test, predictions, average='weighted', zero_division=0)

                        results['Test Size'].append(f'{int(test_size * 100)}%')
                        results['Accuracy'].append(accuracy)
                        results['Recall'].append(recall)
                        results['Training Time'].append(training_time)
                        results['Training Samples'].append(len(X_train))
                        results['Test Samples'].append(len(X_test))

                self._experiment_results['test_size_comparison'] = pd.DataFrame(results)

        def run_clean_vs_normal_comparison(self, test_size: float = 0.2) -> None:
                results = {
                        'Data Type': [],
                        'Accuracy': [],
                        'Recall': [],
                        'Precision': [],
                        'F1 Score': [],
                        'Training Time': []
                }

                data_types = {
                        'Cleaned': 'v2_cleaned_tokenized',
                        'Normal': 'v2_tokenized'
                }

                y = self._data['v1'].to_list()

                for name, column in data_types.items():
                        X = self._data[column].to_list()
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

                        classifier = cl.ClassifierNB(tokenizer=tok.UnigramTokenization)

                        start_time = time.time()
                        classifier.fit(X_train, y_train)
                        training_time = time.time() - start_time

                        predictions = []
                        for sample in X_test:
                                pred_dict = classifier.predict(sample)
                                predictions.append(max(pred_dict, key=pred_dict.get))

                        accuracy = accuracy_score(y_test, predictions)
                        recall = recall_score(y_test, predictions, average='weighted', zero_division=0)
                        precision = precision_score(y_test, predictions, average='weighted', zero_division=0)
                        f1 = f1_score(y_test, predictions, average='weighted', zero_division=0)

                        results['Data Type'].append(name)
                        results['Accuracy'].append(accuracy)
                        results['Recall'].append(recall)
                        results['Precision'].append(precision)
                        results['F1 Score'].append(f1)
                        results['Training Time'].append(training_time)

                self._experiment_results['clean_vs_normal'] = pd.DataFrame(results)

        def generate_confusion_matrix(self, test_size: float = 0.2) -> None:
                X = self._data['v2_cleaned_tokenized'].to_list()
                y = self._data['v1'].to_list()
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

                classifier = cl.ClassifierNB(tokenizer=tok.UnigramTokenization)
                classifier.fit(X_train, y_train)

                predictions = []
                for sample in X_test:
                        pred_dict = classifier.predict(sample)
                        predictions.append(max(pred_dict, key=pred_dict.get))

                cm = confusion_matrix(y_test, predictions)
                labels = sorted(list(set(y)))

                self._experiment_results['confusion_matrix'] = {
                        'matrix': cm,
                        'labels': labels
                }

        def draw_tokenizer_comparison_plot(self, saving_plot: bool = False) -> None:
                if 'tokenizer_comparison' not in self._experiment_results:
                        raise ValueError("Run tokenizer comparison first using run_tokenizer_comparison()")

                sns.set_style("whitegrid")
                sns.set_context("notebook", font_scale=1.1)
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), dpi=100)

                df = self._experiment_results['tokenizer_comparison']
                metrics = ['Accuracy', 'Recall', 'Precision', 'F1 Score']
                x = np.arange(len(df['Tokenizer']))
                width = 0.2

                colors = sns.color_palette("husl", n_colors=len(metrics))

                for i, metric in enumerate(metrics):
                        offset = width * (i - len(metrics) / 2 + 0.5)
                        bars = ax1.bar(
                                x + offset,
                                df[metric],
                                width,
                                label=metric,
                                color=colors[i],
                                edgecolor='white',
                                linewidth=1.5
                        )
                        ax1.bar_label(bars, fmt='%.3f', padding=3, fontsize=9, fontweight='bold')

                ax1.set_title(
                        "Tokenizer Performance Comparison",
                        fontsize=18,
                        fontweight='bold',
                        pad=20,
                        color='#2C3E50'
                )
                ax1.set_xlabel(
                        "Tokenizer Type",
                        fontsize=14,
                        fontweight='semibold',
                        color='#34495E',
                        labelpad=10
                )
                ax1.set_ylabel(
                        "Score",
                        fontsize=14,
                        fontweight='semibold',
                        color='#34495E',
                        labelpad=10
                )
                ax1.set_xticks(x)
                ax1.set_xticklabels(df['Tokenizer'])
                ax1.legend(loc='lower right', frameon=True, shadow=True, fontsize=10)
                ax1.grid(True, axis='y', alpha=0.3, linestyle='--', linewidth=0.7)
                ax1.set_axisbelow(True)
                ax1.tick_params(axis='both', labelsize=11, colors='#34495E')
                ax1.set_facecolor('#FAFAFA')

                bars = ax2.bar(
                        df['Tokenizer'],
                        df['Training Time'],
                        color=sns.color_palette("husl", n_colors=len(df)),
                        edgecolor='white',
                        linewidth=2
                )
                ax2.bar_label(bars, fmt='%.4f s', padding=5, fontsize=11, fontweight='bold')

                ax2.set_title(
                        "Training Time Comparison",
                        fontsize=18,
                        fontweight='bold',
                        pad=20,
                        color='#2C3E50'
                )
                ax2.set_xlabel(
                        "Tokenizer Type",
                        fontsize=14,
                        fontweight='semibold',
                        color='#34495E',
                        labelpad=10
                )
                ax2.set_ylabel(
                        "Time (seconds)",
                        fontsize=14,
                        fontweight='semibold',
                        color='#34495E',
                        labelpad=10
                )
                ax2.grid(True, axis='y', alpha=0.3, linestyle='--', linewidth=0.7)
                ax2.set_axisbelow(True)
                ax2.tick_params(axis='both', labelsize=11, colors='#34495E')
                ax2.set_facecolor('#FAFAFA')

                sns.despine(left=False, bottom=False, top=True, right=True)
                fig.patch.set_facecolor('white')
                plt.tight_layout()

                if saving_plot:
                        plt.savefig(
                                f'{self._storage}/TokenizerComparison.png',
                                dpi=300,
                                bbox_inches='tight',
                                facecolor='white',
                                edgecolor='none'
                        )

                plt.show()

        def draw_test_size_comparison_plot(self, saving_plot: bool = False) -> None:
                if 'test_size_comparison' not in self._experiment_results:
                        raise ValueError("Run test size comparison first using run_test_size_comparison()")

                sns.set_style("whitegrid")
                sns.set_context("notebook", font_scale=1.1)
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), dpi=100)

                df = self._experiment_results['test_size_comparison']
                x = np.arange(len(df['Test Size']))

                ax1.plot(x, df['Accuracy'], marker='o', linewidth=2.5, markersize=8, label='Accuracy', color='#3498db')
                ax1.plot(x, df['Recall'], marker='s', linewidth=2.5, markersize=8, label='Recall', color='#e74c3c')

                for i, (acc, rec) in enumerate(zip(df['Accuracy'], df['Recall'])):
                        ax1.text(i, acc, f'{acc:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
                        ax1.text(i, rec, f'{rec:.3f}', ha='center', va='top', fontsize=9, fontweight='bold')

                ax1.set_title(
                        "Accuracy and Recall vs Test Size",
                        fontsize=18,
                        fontweight='bold',
                        pad=20,
                        color='#2C3E50'
                )
                ax1.set_xlabel(
                        "Test Size",
                        fontsize=14,
                        fontweight='semibold',
                        color='#34495E',
                        labelpad=10
                )
                ax1.set_ylabel(
                        "Score",
                        fontsize=14,
                        fontweight='semibold',
                        color='#34495E',
                        labelpad=10
                )
                ax1.set_xticks(x)
                ax1.set_xticklabels(df['Test Size'])
                ax1.legend(loc='best', frameon=True, shadow=True, fontsize=11)
                ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.7)
                ax1.set_axisbelow(True)
                ax1.tick_params(axis='both', labelsize=11, colors='#34495E')
                ax1.set_facecolor('#FAFAFA')

                bars = ax2.bar(
                        df['Test Size'],
                        df['Training Time'],
                        color=sns.color_palette("husl", n_colors=len(df)),
                        edgecolor='white',
                        linewidth=2
                )
                ax2.bar_label(bars, fmt='%.4f s', padding=5, fontsize=11, fontweight='bold')

                ax2.set_title(
                        "Training Time vs Test Size",
                        fontsize=18,
                        fontweight='bold',
                        pad=20,
                        color='#2C3E50'
                )
                ax2.set_xlabel(
                        "Test Size",
                        fontsize=14,
                        fontweight='semibold',
                        color='#34495E',
                        labelpad=10
                )
                ax2.set_ylabel(
                        "Time (seconds)",
                        fontsize=14,
                        fontweight='semibold',
                        color='#34495E',
                        labelpad=10
                )
                ax2.grid(True, axis='y', alpha=0.3, linestyle='--', linewidth=0.7)
                ax2.set_axisbelow(True)
                ax2.tick_params(axis='both', labelsize=11, colors='#34495E')
                ax2.set_facecolor('#FAFAFA')

                sns.despine(left=False, bottom=False, top=True, right=True)
                fig.patch.set_facecolor('white')
                plt.tight_layout()

                if saving_plot:
                        plt.savefig(
                                f'{self._storage}/TestSizeComparison.png',
                                dpi=300,
                                bbox_inches='tight',
                                facecolor='white',
                                edgecolor='none'
                        )

                plt.show()

        def draw_clean_vs_normal_plot(self, saving_plot: bool = False) -> None:
                if 'clean_vs_normal' not in self._experiment_results:
                        raise ValueError("Run clean vs normal comparison first using run_clean_vs_normal_comparison()")

                sns.set_style("whitegrid")
                sns.set_context("notebook", font_scale=1.1)
                fig, ax = plt.subplots(figsize=(12, 7), dpi=100)

                df = self._experiment_results['clean_vs_normal']
                metrics = ['Accuracy', 'Recall', 'Precision', 'F1 Score']
                x = np.arange(len(df['Data Type']))
                width = 0.2

                colors = sns.color_palette("husl", n_colors=len(metrics))

                for i, metric in enumerate(metrics):
                        offset = width * (i - len(metrics) / 2 + 0.5)
                        bars = ax.bar(
                                x + offset,
                                df[metric],
                                width,
                                label=metric,
                                color=colors[i],
                                edgecolor='white',
                                linewidth=1.5
                        )
                        ax.bar_label(bars, fmt='%.3f', padding=3, fontsize=9, fontweight='bold')

                ax.set_title(
                        "Cleaned vs Normal Data Performance",
                        fontsize=18,
                        fontweight='bold',
                        pad=20,
                        color='#2C3E50'
                )
                ax.set_xlabel(
                        "Data Type",
                        fontsize=14,
                        fontweight='semibold',
                        color='#34495E',
                        labelpad=10
                )
                ax.set_ylabel(
                        "Score",
                        fontsize=14,
                        fontweight='semibold',
                        color='#34495E',
                        labelpad=10
                )
                ax.set_xticks(x)
                ax.set_xticklabels(df['Data Type'])
                ax.legend(loc='lower right', frameon=True, shadow=True, fontsize=11)
                ax.grid(True, axis='y', alpha=0.3, linestyle='--', linewidth=0.7)
                ax.set_axisbelow(True)
                sns.despine(left=False, bottom=False, top=True, right=True)
                ax.tick_params(axis='both', labelsize=11, colors='#34495E')
                ax.set_facecolor('#FAFAFA')
                fig.patch.set_facecolor('white')

                plt.tight_layout()

                if saving_plot:
                        plt.savefig(
                                f'{self._storage}/CleanVsNormal.png',
                                dpi=300,
                                bbox_inches='tight',
                                facecolor='white',
                                edgecolor='none'
                        )

                plt.show()

        def draw_confusion_matrix_plot(self, saving_plot: bool = False) -> None:
                if 'confusion_matrix' not in self._experiment_results:
                        raise ValueError("Generate confusion matrix first using generate_confusion_matrix()")

                sns.set_style("white")
                sns.set_context("notebook", font_scale=1.1)
                fig, ax = plt.subplots(figsize=(10, 8), dpi=100)

                cm = self._experiment_results['confusion_matrix']['matrix']
                labels = self._experiment_results['confusion_matrix']['labels']

                sns.heatmap(
                        cm,
                        annot=True,
                        fmt='d',
                        cmap='Blues',
                        xticklabels=labels,
                        yticklabels=labels,
                        ax=ax,
                        cbar_kws={'label': 'Count'},
                        linewidths=2,
                        linecolor='white',
                        square=True
                )

                ax.set_title(
                        "Confusion Matrix",
                        fontsize=18,
                        fontweight='bold',
                        pad=20,
                        color='#2C3E50'
                )
                ax.set_xlabel(
                        "Predicted Label",
                        fontsize=14,
                        fontweight='semibold',
                        color='#34495E',
                        labelpad=10
                )
                ax.set_ylabel(
                        "True Label",
                        fontsize=14,
                        fontweight='semibold',
                        color='#34495E',
                        labelpad=10
                )

                ax.tick_params(axis='both', labelsize=11, colors='#34495E')
                fig.patch.set_facecolor('white')

                plt.tight_layout()

                if saving_plot:
                        plt.savefig(
                                f'{self._storage}/ConfusionMatrix.png',
                                dpi=300,
                                bbox_inches='tight',
                                facecolor='white',
                                edgecolor='none'
                        )

                plt.show()

        def draw_training_time_comparison_plot(self, saving_plot: bool = False) -> None:
                if 'clean_vs_normal' not in self._experiment_results:
                        raise ValueError("Run clean vs normal comparison first using run_clean_vs_normal_comparison()")

                sns.set_style("whitegrid")
                sns.set_context("notebook", font_scale=1.1)
                fig, ax = plt.subplots(figsize=(12, 7), dpi=100)

                df = self._experiment_results['clean_vs_normal']

                bars = ax.bar(
                        df['Data Type'],
                        df['Training Time'],
                        color=sns.color_palette("husl", n_colors=len(df)),
                        edgecolor='white',
                        linewidth=2
                )
                ax.bar_label(bars, fmt='%.4f s', padding=5, fontsize=11, fontweight='bold')

                ax.set_title(
                        "Training Time: Cleaned vs Normal Data",
                        fontsize=18,
                        fontweight='bold',
                        pad=20,
                        color='#2C3E50'
                )
                ax.set_xlabel(
                        "Data Type",
                        fontsize=14,
                        fontweight='semibold',
                        color='#34495E',
                        labelpad=10
                )
                ax.set_ylabel(
                        "Training Time (seconds)",
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
                                f'{self._storage}/TrainingTimeComparison.png',
                                dpi=300,
                                bbox_inches='tight',
                                facecolor='white',
                                edgecolor='none'
                        )

                plt.show()
