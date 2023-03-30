from utils import get_data, plot_metrics, normalize
from model import MultiClassSVM, PCA
from typing import Tuple
from main import get_hyperparameters
from matplotlib import pyplot as plt
import numpy as np


def plot_debug_metrics(debug_metrics) -> Tuple[float, int, float]:
    """
    Plot the metrics of each individual model in Multi-Class SVM.
    """
    ks, accuracies, precisions, recalls, f1_scores = zip(*debug_metrics)
    fig, ax = plt.subplots(figsize=(15, 10))
    
    font = {'family' : 'serif',
            'size'   : 20,
            'serif':  'cmr10'
            }

    plt.rc('font', **font)

    ax.plot(ks, accuracies, marker='o', label='Accuracy')
    ax.plot(ks, precisions, marker='o', label='Precision')
    ax.plot(ks, recalls, marker='o', label='Recall')
    ax.plot(ks, f1_scores, marker='o', label='F1-score')
    ax.set_xticks(ks)
    ax.set_xticklabels(ks)
    plt.yticks(fontfamily="serif", fontsize=16)
    plt.xticks(fontfamily="serif", fontsize=16)
    ax.set_xlabel('Model for Digit', fontfamily="serif", fontsize=20.)
    ax.set_ylabel('Value of Metric', fontfamily="serif", fontsize=20)
    ax.set_title('Individual Model performance', fontfamily="serif", fontsize=20)
    ax.legend()
    ax.yaxis.set_ticks(np.arange(0, 1.05, 0.05))
    plt.grid()
    plt.savefig('debug_metrics.png')


def main() -> None:
    # hyperparameters
    learning_rate, num_iters, C = get_hyperparameters()

    # get data
    X_train, X_test, y_train, y_test = get_data()

    # normalize the data
    X_train, X_test = normalize(X_train, X_test)

    # create a model
    svm = MultiClassSVM(num_classes=10)

    # fit the model
    svm.fit(
        X_train, y_train, C=C,
        learning_rate=learning_rate,
        num_iters=num_iters,
    )
    # get debug metrics
    debug_metrics = svm.debug(X_test, y_test)

    # plot and save the results
    plot_debug_metrics(debug_metrics)


if __name__ == '__main__':
    main()
