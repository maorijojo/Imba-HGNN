import numpy as np
import scipy
import random
import torch
import scipy.sparse as sp

def process_result(arr: np.ndarray):
    arr_mean = np.mean(arr)
    arr_std = np.std(arr, ddof=1)
    return "{:.5f} ± {:.5f}".format(arr_mean, arr_std)

def get_meta_path_adjacency_matrix(adj, type_mask, metapath):
    out_adj = scipy.sparse.csr_matrix(adj.toarray()[np.ix_(type_mask == metapath[0], type_mask == metapath[1])])
    for i in range(1, len(metapath) - 1):
        out_adj = out_adj.dot(
            scipy.sparse.csr_matrix(adj.toarray()[np.ix_(type_mask == metapath[i], type_mask == metapath[i + 1])])
        )
    return out_adj.toarray()


def extend_mx(mx, dim, pad=0):
    N = mx.shape[0]
    mx_1 = np.ones(shape=(N, dim)) * pad
    mx_2 = np.ones(shape=(dim, N+dim)) * pad
    mx = np.hstack((mx, mx_1))
    mx = np.vstack((mx, mx_2))
    return mx


def judge_meta_path(meta_path_list, labels, train_idx):
    res = list()
    for meta_path in meta_path_list:
        row, col = np.nonzero(meta_path)
        same = 0
        diff = 0
        for r, c in zip(row, col):
            if r in train_idx and c in train_idx:
                if labels[r] == labels[c]:
                    same += 1
                else:
                    diff += 1
        if same == 0 and diff == 0:
            res.append(0)
        else:
            res.append(same/(same+diff))
    return res



def _generate_fake_features(target_fea, idx_real, num_fake, adj, shift, p_sigma):
    real_features = target_fea[idx_real]
    center = real_features.sum(axis=0) / real_features.shape[0]
    fake_features = []
    num_real = idx_real.shape[0]
    dis_min = np.inf
    dis_max = -1
    for fea in real_features:
        dis = np.linalg.norm(fea - center)
        dis_min = min(dis_min, dis)
        dis_max = max(dis_max, dis)

    for i in range(num_fake):
        unit_vec = np.random.random(size=center.shape)
        unit_vec = unit_vec - 0.5
        norm = np.linalg.norm(unit_vec)
        if norm != 0:
            unit_vec = unit_vec / norm
        sigma = (dis_max - dis_min) / p_sigma
        mu = (dis_max + dis_min) / 2
        ratio = random.gauss(mu, sigma)

        fake_fea = center + ratio * unit_vec
        fake_features.append(fake_fea)
        dist_min, real_node_chosen = np.inf, -1
        for idx_r in idx_real:
            dist = np.linalg.norm(target_fea[idx_r] - fake_fea)
            if dist < dist_min:
                dist_min = dist
                real_node_chosen = idx_r
    fake_features = np.array(fake_features)
    return fake_features


def SMOTE(target_fea, label2idx_train, labels, idx_data, adj, p_sigma, fake_ratio):
    max_sample = -1
    sample_list = []
    for item in label2idx_train.items():
        sample_list.append(item[1].shape[0])
        max_sample = max(max_sample, item[1].shape[0])

    fake_nodes = 0
    for sample in sample_list:
        fake_nodes += int((max_sample-sample) * fake_ratio)

    real_target_num = labels.shape[0]
    cur_num = real_target_num 
    real_nodes = adj.shape[0]
    shift = real_nodes
    label2idx_fake = dict()

    fake_fea = []
    meta_path_mask = np.zeros(shape=(real_target_num+fake_nodes, real_target_num+fake_nodes), dtype=bool)
    for item in label2idx_train.items():
        num_fake = int((max_sample - item[1].shape[0]) * fake_ratio)
        idx_real = item[1]
        print("第%d类需要生成%d个fake nodes" % (item[0], num_fake))
        if num_fake == 0:
            continue
        fake_features = _generate_fake_features(target_fea, idx_real, num_fake, adj, shift, p_sigma)
        fake_fea.append(fake_features)
        shift += num_fake

        labels = np.concatenate((labels, np.repeat(item[0], num_fake)))
        label2idx_fake[item[0]] = np.arange(cur_num, cur_num + num_fake)

        for i in range(cur_num, cur_num+num_fake):
            for j in idx_real:
                meta_path_mask[i][j] = True

        cur_num += num_fake

    idx_data['train_idx'] = np.concatenate((idx_data['train_idx'], np.arange(real_target_num, cur_num)))
    fake_fea = np.vstack(fake_fea)
    print("fake_fea.shape: ", fake_fea.shape)

    return fake_fea, labels, idx_data, label2idx_fake, meta_path_mask


