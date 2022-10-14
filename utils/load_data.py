import sys
sys.path.append(r'../')
import os
import torch
import numpy as np
import scipy
from queue import Queue
import scipy.sparse as sp
import scipy.io as sio
from collections import defaultdict, OrderedDict
from utils.utils import *

def load_data(dataset: str):
    if dataset == "acm":
        return load_data_ACM()
    if dataset == "dblp":
        return load_data_DBLP()
    if dataset == "imdb":
        return load_data_IMDB()
    print("Dataset does not exist!")
    raise ValueError()


def load_data_ACM():
    path = './data/ACM'
    adj = scipy.sparse.load_npz(path + '/adjM.npz').tocoo()
    features_0 = np.load(path + '/features_0.npy')
    features_1 = np.load(path + '/features_1.npy')
    features_2 = np.load(path + '/features_2.npy')
    labels = np.load(os.path.join(path, "labels.npy"))
    node_types = np.load(os.path.join(path, "node_types.npy"))
    train_val_test_idx_data = np.load(
        os.path.join(path, "train_val_test_idx.npz"))  # ['train_idx', 'val_idx', 'test_idx']
    target_type = 0

    features_list = [features_0, features_1, features_2]  # [ndarray,...]

    labels_set = set(labels)
    num_classes = len(labels_set)
    labels_one_hot = np.zeros(shape=(labels.shape[0], num_classes))
    for i, la in enumerate(labels):
        labels_one_hot[i][la] = 1

    train_val_test_idx_data = {
        "train_idx": train_val_test_idx_data["train_idx"],
        "val_idx": train_val_test_idx_data["val_idx"],
        "test_idx": train_val_test_idx_data["test_idx"]
    }

    train_idx = train_val_test_idx_data["train_idx"]
    label2idx_train = dict()
    for idx in train_idx:
        label = labels[idx]
        if label not in label2idx_train.keys():
            label2idx_train[label] = []
        label2idx_train[label].append(idx)
    for key in label2idx_train.keys():
        label2idx_train[key] = np.array(label2idx_train[key], dtype=int)

    type_mask = []
    for i, cnt in enumerate(node_types):
        type_mask.extend([i] * cnt)
    type_mask = np.array(type_mask, dtype=int)

    meta_paths = [[0, 1, 0], [0, 2, 0]]
    meta_path_list = []
    for meta in meta_paths:
        meta_path_list.append(get_meta_path_adjacency_matrix(adj, type_mask, meta))

    return features_list, adj, meta_path_list, labels, labels_one_hot, train_val_test_idx_data, \
           num_classes, label2idx_train, type_mask, target_type


def load_data_DBLP():
    path = './data/DBLP'

    features_0 = np.load(path + '/features_0.npy')
    features_1 = np.load(path + '/features_1.npy')
    features_2 = np.load(path + '/features_2.npy')
    features_list = [features_0, features_1, features_2]
    target_type = 0

    adj = scipy.sparse.load_npz(path + "/adjM.npz").tocoo()

    labels = np.load(path + "/labels.npy")
    labels_set = set(labels)
    num_classes = len(labels_set)
    labels_one_hot = np.zeros(shape=(labels.shape[0], num_classes))
    for i, la in enumerate(labels):
        labels_one_hot[i][la] = 1

    train_val_test_idx_data = np.load(path + "/train_val_test_idx.npz")
    train_val_test_idx_data = {
        "train_idx": train_val_test_idx_data["train_idx"],
        "val_idx": train_val_test_idx_data["val_idx"],
        "test_idx": train_val_test_idx_data["test_idx"]
    }

    train_idx = train_val_test_idx_data["train_idx"]
    label2idx_train = dict()
    for idx in train_idx:
        label = labels[idx]
        if label not in label2idx_train.keys():
            label2idx_train[label] = []
        label2idx_train[label].append(idx)
    for key in label2idx_train.keys():
        label2idx_train[key] = np.array(label2idx_train[key], dtype=int)

    node_types = np.load(path + "/node_types.npy")
    type_mask = []
    for i, cnt in enumerate(node_types):
        type_mask.extend([i] * cnt)
    type_mask = np.array(type_mask, dtype=int)

    meta_paths = [[0, 1, 0], [0, 1, 2, 1, 0]]
    meta_path_list = []
    for meta in meta_paths:
        meta_path_list.append(get_meta_path_adjacency_matrix(adj, type_mask, meta))

    return features_list, adj, meta_path_list, labels, labels_one_hot, train_val_test_idx_data, \
           num_classes, label2idx_train, type_mask, target_type


def load_data_IMDB():
    path = './data/IMDB'
    features_0 = np.load(path + '/features_0.npy')
    features_1 = np.load(path + '/features_1.npy')
    features_2 = np.load(path + '/features_2.npy')
    adj = scipy.sparse.load_npz(path + '/adjM.npz').tocoo()
    node_types = np.load(path + '/node_types.npy')
    labels = np.load(path + '/labels.npy')
    train_val_test_idx_data = np.load(path + '/train_val_test_idx.npz')
    target_type = 0

    features_list = [features_0, features_1, features_2]
    for fea in features_list:
        print(fea.shape)

    labels_set = set(labels)
    num_classes = len(labels_set)
    labels_one_hot = np.zeros(shape=(labels.shape[0], num_classes))
    for i, la in enumerate(labels):
        labels_one_hot[i][la] = 1

    train_val_test_idx_data = {
        "train_idx": train_val_test_idx_data["train_idx"],
        "val_idx": train_val_test_idx_data["val_idx"],
        "test_idx": train_val_test_idx_data["test_idx"]
    }

    train_idx = train_val_test_idx_data["train_idx"]
    label2idx_train = dict()
    for idx in train_idx:
        label = labels[idx]
        if label not in label2idx_train.keys():
            label2idx_train[label] = []
        label2idx_train[label].append(idx)
    for key in label2idx_train.keys():
        label2idx_train[key] = np.array(label2idx_train[key], dtype=int)

    type_mask = []
    for i, cnt in enumerate(node_types):
        type_mask.extend([i] * cnt)
    type_mask = np.array(type_mask, dtype=int)

    meta_paths = [[0, 1, 0], [0, 2, 0]]
    meta_path_list = []
    for meta in meta_paths:
        meta_path_list.append(get_meta_path_adjacency_matrix(adj, type_mask, meta))

    return features_list, adj, meta_path_list, labels, labels_one_hot, train_val_test_idx_data, num_classes, \
           label2idx_train, type_mask, target_type


def load_data_AIFB():
    path = './data/AIFB'
    adj = scipy.sparse.load_npz(path + '/adjM.npz').tocoo()
    labels = np.load(path + '/labels.npy')
    train_val_test_idx_data = np.load(path + "/train_val_test_idx.npz")
    node_types = np.load(path + "/node_types.npy")
    target_type = 0

    features_list = []
    for i in range(7):
        features_list.append(np.load(path + '/features_{}.npy'.format(i)))

    labels_set = set(labels)
    num_classes = 4
    labels_one_hot = np.zeros(shape=(labels.shape[0], num_classes))
    for i, la in enumerate(labels):
        if la == -1:
            continue
        labels_one_hot[i][la] = 1

    train_val_test_idx_data = {
        "train_idx": train_val_test_idx_data["train_idx"],
        "val_idx": train_val_test_idx_data["val_idx"],
        "test_idx": train_val_test_idx_data["test_idx"]
    }

    train_idx = train_val_test_idx_data["train_idx"]
    label2idx_train = dict()
    for idx in train_idx:
        label = labels[idx]
        if label not in label2idx_train.keys():
            label2idx_train[label] = []
        label2idx_train[label].append(idx)
    for key in label2idx_train.keys():
        label2idx_train[key] = np.array(label2idx_train[key], dtype=int)

    type_mask = node_types

    meta_paths = [[0, 1, 0], [0, 2, 0], [0, 3, 0], [0, 4, 0], [0, 5, 0], [0, 6, 0]]
    meta_path_list = []
    for meta in meta_paths:
        meta_path_list.append(get_meta_path_adjacency_matrix(adj, type_mask, meta))

    return features_list, adj, meta_path_list, labels, labels_one_hot, train_val_test_idx_data, \
           num_classes, label2idx_train, type_mask, target_type

