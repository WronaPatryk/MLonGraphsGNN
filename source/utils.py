import copy
import itertools
import random
import tensorflow as tf
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def set_seeds(seed):
    tf.random.set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def cut_samples_to_n(samples, n=1600):
    """ Cut the audio file to first 1600 values. If the vector is longer than 1600, fill with 0."""
    samples_len = len(samples)
    if samples_len  == n:
        return samples
    elif samples_len  > n:
        return samples[0:n]
    else:
        no_missing = n - samples_len 
        zeros = [0] * no_missing
        return np.concatenate((samples,zeros))
    
# means of rows of a spectrogram
def s_vectorize2(s):
    return np.mean(s, axis=1)

# means of values inside each of 4x4 windows
def s_vectorize3(s):
    window_step_x = 4
    window_step_y = 4
    output_list = []
    for i in range(0,s.shape[0],window_step_x):
        for j in range(0,s.shape[1],window_step_y):
            output_list.append(np.mean(s[i:(i+4),j:(j+4)]))
            
    return np.array(output_list) 

def train_val_test_lists(dir, train_file, valid_file, test_file, header=None, index_col=None):

    train_list = pd.read_csv(dir + train_file, header=header, index_col=index_col).iloc[:,0].values.tolist()
    test_list = pd.read_csv(dir + valid_file,  header=header, index_col=index_col).iloc[:,0].values.tolist()
    valid_list = pd.read_csv(dir + test_file, header=header, index_col=index_col).iloc[:,0].values.tolist()

    return train_list, test_list, valid_list

def train_val_test_labels(dir, train_file, valid_file, test_file, header=None, index_col=None, pos = 0):
    """ for raw_files, pos = 0, for spectrograms pos=1"""
    train_list, test_list, valid_list = train_val_test_lists(dir, train_file, valid_file, test_file, header, index_col)

    labels_train = []

    for path in train_list:
        labels_train.append(path.split("/")[pos])
        
    labels_valid = []

    for path in valid_list:
        labels_valid.append(path.split("/")[pos])

    labels_test = []

    for path in test_list:
        labels_test.append(path.split("/")[pos])


    return labels_train, labels_valid, labels_test


def labels_to_numbers(labels):
    return np.unique(labels, return_inverse=True)

def visualize(h, color):
    z = TSNE(n_components=2).fit_transform(h.detach().cpu().numpy())

    plt.figure(figsize=(10,10))
    plt.xticks([])
    plt.yticks([])

    plt.scatter(z[:, 0], z[:, 1], s=70, c=color, cmap="Set2")
    plt.show()

def load_data_df(data_dir, header=None):
    train_df = pd.read_csv(data_dir+"train.csv", header=header)
    valid_df = pd.read_csv(data_dir+"valid.csv", header=header)
    test_df = pd.read_csv(data_dir+"test.csv", header=header)
    df = pd.concat([train_df, valid_df, test_df])
    return df

def describe_pyg_data(data):
    print(f'Number of nodes: {data.num_nodes}')
    print(f'Number of edges: {data.num_edges}')
    print(f'Number of features: {data.num_features}')
    print(f'Number of classes: {data.num_classes}')
    print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
    print(f'Number of training nodes: {data.train_mask.sum()}')
    print(f'Number of validation nodes: {data.valid_mask.sum()}')
    print(f'Number of testing nodes: {data.test_mask.sum()}')
    print(f'Training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.2f}')
    print(f'Has isolated nodes: {data.has_isolated_nodes()}')
    print(f'Has self-loops: {data.has_self_loops()}')
    print(f'Is undirected: {data.is_undirected()}')