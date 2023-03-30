import numpy as np
import pandas as pd
import os
from typing import Tuple
from matplotlib import pyplot as plt

def get_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # load the data
    if os.path.exists('data/X_train.csv') and  os.path.exists('data/y_train.csv') and  os.path.exists('data/X_test.csv') and  os.path.exists('data/y_test.csv'):
        X_train = pd.read_csv('data/X_train.csv')
        y_train = pd.read_csv('data/y_train.csv')
        X_test = pd.read_csv('data/X_test.csv')
        y_test = pd.read_csv('data/y_test.csv')
        
        return X_train.values, X_test.values, y_train.iloc[:, 0].values, y_test.iloc[:, 0].values
    else:
        from keras.datasets import mnist
        (X_train, y_train), (X_test, y_test) = mnist.load_data()

        X_train = X_train.reshape((X_train.shape[0], 28*28)).astype('float32')
        X_test = X_test.reshape((X_test.shape[0], 28*28)).astype('float32')
        
        pd.DataFrame(X_train).to_csv('data/X_train.csv', index = False)
        pd.DataFrame(y_train).to_csv('data/y_train.csv', index = False)
        pd.DataFrame(X_test).to_csv('data/X_test.csv', index = False)
        pd.DataFrame(y_test).to_csv('data/y_test.csv', index = False)
        

    return X_train, X_test, y_train, y_test


def normalize(X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

    X_train_norm = (2*X_train/255) -1
    X_test_norm = (2*X_test/255) -1
    
    return X_train_norm, X_test_norm


def plot_metrics(metrics):
    """
    Plot the metrics against the number of PCA components.
    """
    ks, accuracies, precisions, recalls, f1_scores = zip(*metrics)
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
    ax.set_xscale('log')
    ax.set_xticks(ks)
    ax.set_xticklabels(ks)
    plt.yticks(fontfamily="serif", fontsize=16)
    plt.xticks(fontfamily="serif", fontsize=16)
    ax.set_xlabel('Number of PCA components', fontfamily="serif", fontsize=20.)
    ax.set_ylabel('Value of Metric', fontfamily="serif", fontsize=20)
    ax.set_title('Model performance', fontfamily="serif", fontsize=20)
    ax.legend()
    ax.yaxis.set_ticks(np.arange(0.5, 1.05, 0.05))
    plt.grid()
    plt.savefig('metrics.png')