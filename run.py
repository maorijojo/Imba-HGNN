import sys
import os
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score, f1_score
from utils.utils import *
from utils.load_data import *
from models.ImbaHgnn import *

sys.path.append(r'./')

# Training setting
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='acm')
parser.add_argument('--sparse', action='store_true', default=False)
parser.add_argument('--repeats', type=int, default=5)
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--weight_decay', type=float, default=0.001)
parser.add_argument('--hid_dim', type=int, default=32)
parser.add_argument('--n_hid', type=int, default=8)
parser.add_argument('--n_heads', type=int, default=4)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--alpha', type=float, default=1)
parser.add_argument('--beta', type=float, default=1)
parser.add_argument('--patience', type=int, default=100)
parser.add_argument('--sigma', type=float, default=6)
parser.add_argument('--fake_ratio', type=float, default=1)

args = parser.parse_args()

model = "Imba-HGNN"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device: ", device, ", Dataset: ", args.dataset, ", Model: ", model, ", Sparse: ", args.sparse)

features_list, adj, meta_path_list, labels, labels_one_hot, \
idx_data, num_classes, label2idx_train, type_mask, target_type = load_data(args.dataset)

train_idx, val_idx, test_idx = idx_data["train_idx"], idx_data["val_idx"], idx_data["test_idx"]

valid_meta_path_list = list()
valid_ratio_new = list()
for i in range(len(valid_ratio)):
    if valid_ratio[i] != 0:
        valid_meta_path_list.append(meta_path_list[i])
        valid_ratio_new.append(valid_ratio[i])
meta_path_list = valid_meta_path_list
valid_ratio = valid_ratio_new

for i, meta_path in enumerate(meta_path_list):
    row, col = np.where(meta_path > 1)
    for r, c in zip(row, col):
        meta_path_list[i][r][c] = 1

in_dims = [fea.shape[1] for fea in features_list]
all_nodes = adj.shape[0]
target_type_mask = (type_mask == target_type)

target_fea = features_list[target_type]
fake_fea, labels, idx_data, label2idx_fake, meta_path_mask = SMOTE(target_fea, label2idx_train, labels,
                                                                       idx_data, adj, args.sigma, args.fake_ratio)
features_list.append(fake_fea)
fake_type = len(features_list) - 1
fake_nodes = fake_fea.shape[0]
real_nodes = all_nodes
all_nodes = real_nodes + fake_nodes
real_mask = np.concatenate((np.repeat(True, real_nodes), np.repeat(False, fake_nodes)))
target_in_real_mask = target_type_mask
target_type_mask = np.concatenate((target_type_mask, np.repeat(True, fake_nodes)))
target_features = np.vstack((target_fea, fake_fea))
real_in_target_mask = np.concatenate((np.repeat(True, target_fea.shape[0]), np.repeat(False, fake_nodes)))

for i in range(len(meta_path_list)):
    meta_path_list[i] = extend_mx(meta_path_list[i], fake_nodes, 0)

if args.sparse:
    row, col, val = adj.row, adj.col, adj.data
    index = torch.from_numpy(np.vstack((row, col))).long()
    val = torch.from_numpy(val).float()
    adj = torch.sparse.FloatTensor(index, val, torch.Size(adj.shape)).to(device)
else:
    adj = torch.from_numpy(adj.A).to(device)

features_list = [torch.from_numpy(fea).float().to(device) for fea in features_list]

labels = torch.LongTensor(labels).to(device)
train_idx = idx_data['train_idx']
val_idx = idx_data['val_idx']
test_idx = idx_data['test_idx']

meta_path_list = [torch.from_numpy(meta_path).float().to(device) for meta_path in meta_path_list]
meta_path_mask = torch.from_numpy(meta_path_mask).to(device)
target_features = torch.from_numpy(target_features).float().to(device)

label_val_list = labels[val_idx].cpu().numpy().tolist()
label_test_list = labels[test_idx].cpu().numpy().tolist()

labels_val_one_hot = labels_one_hot[val_idx].tolist()
labels_test_one_hot = labels_one_hot[test_idx].tolist()

res = []
res_acc = np.empty(shape=(args.repeats,), dtype=np.float32)
res_macro = np.empty(shape=(args.repeats,), dtype=np.float32)
res_roc = np.empty(shape=(args.repeats,), dtype=np.float32)

class_acc = np.empty(shape=(num_classes, args.repeats), dtype=np.float32)

for i in range(args.repeats):
    print("=============== {} ===============".format(i + 1))
        
    bceloss = nn.BCEWithLogitsLoss()
    net = ImbaHgnn(in_dims, adj, meta_path_list, args.hid_dim, args.n_hid, num_classes, args.dropout,
                    args.n_heads, target_type_mask, real_mask, target_in_real_mask, real_in_target_mask,
                    meta_path_mask, valid_ratio, sparse=args.sparse)

    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # early stop
    loss_val_min = np.inf
    macro_val_max = 0.0
    acc_val_max = 0.0
    curr_step = 0
    save_path = r'./params/{}_{}.pkl'.format(model, args.dataset)

    time_start_train = time.time()
    epoch = 0
    for epoch in range(args.epochs):
        net.train()
        meta_path_preds, out, loss_embed = net(features_list, target_features)
        loss_fea2mp_list = []
        for pred_adj, label_adj in zip(meta_path_preds, meta_path_list):
            pred_adj_real = pred_adj[real_in_target_mask, :][:, real_in_target_mask]
            label_adj_real = label_adj[real_in_target_mask, :][:, real_in_target_mask]
            loss_fea2mp_list.append(bceloss(pred_adj_real, label_adj_real))
        loss_fea2mp = sum(loss_fea2mp_list)
        loss_GNN = F.nll_loss(out[train_idx], labels[train_idx])
        loss_train = loss_GNN + args.alpha * loss_fea2mp + args.beta * loss_embed

        _, preds_train = torch.max(out[train_idx], dim=1)
        acc_train = torch.sum(preds_train == labels[train_idx]).to(torch.float32) / train_idx.shape[0]

        # auto grad
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()

        # validation
        net.eval()
        with torch.no_grad():
            meta_path_preds, out, loss_embed = net(features_list, target_features)
            loss_fea2mp_list = []
            for pred_adj, label_adj in zip(meta_path_preds, meta_path_list):
                pred_adj_real = pred_adj[real_in_target_mask, :][:, real_in_target_mask]
                label_adj_real = label_adj[real_in_target_mask, :][:, real_in_target_mask]
                loss_fea2mp_list.append(bceloss(pred_adj_real, label_adj_real))
            loss_fea2mp_val = sum(loss_fea2mp_list)
            loss_GNN_val = F.nll_loss(out[val_idx], labels[val_idx])
            loss_val = loss_GNN_val + args.alpha * loss_fea2mp_val + args.beta * loss_embed

            _, preds_val = torch.max(out[val_idx], dim=1)
            acc_val = torch.sum(preds_val == labels[val_idx]).to(torch.float32) / val_idx.shape[0]

            preds_val_list = preds_val.cpu().numpy().tolist()
            macro_f1 = f1_score(label_val_list, preds_val_list, average='macro')

            prediction_val = out[val_idx].cpu().numpy().tolist()
            roc = roc_auc_score(labels_val_one_hot, prediction_val, multi_class='ovo')

        print('Epoch:{:05d} | Train_loss:{:.4f} | Train_acc:{:.4f} | Val_loss:{:.4f} | Val_acc:{:.4f} | '
              'Val_macro_f1:{:.4f} | Val_ROC:{:.4f}'
              .format(epoch, loss_train.item(), acc_train, loss_val.item(), acc_val, macro_f1, roc),
              end=" | ")
        print("loss_fea2mp:{:.4f} | loss_GNN:{:.4f} | loss_embed:{:.4f}".format(loss_fea2mp, loss_GNN, loss_embed))
        loss_val = loss_val.item()
        acc_val = acc_val.item()
        if loss_val <= loss_val_min or acc_val >= acc_val_max: 
            if loss_val <= loss_val_min and acc_val >= acc_val_max:
                torch.save(net.state_dict(), save_path)
            loss_val_min = min(loss_val_min, loss_val)
            acc_val_max = max(acc_val_max, acc_val)
            curr_step = 0
        else:
            curr_step += 1
            if curr_step == args.patience:
                print("Early Stop!")
                break

    time_end_train = time.time()

    net.load_state_dict(torch.load(save_path))
    net.eval()
    time_start_test = time.time()
    with torch.no_grad():
        meta_path_preds, out, loss_embed = net(features_list, target_features)
        loss_fea2mp_list = []
        for pred_adj, label_adj in zip(meta_path_preds, meta_path_list):
            pred_adj_real = pred_adj[real_in_target_mask, :][:, real_in_target_mask]
            label_adj_real = label_adj[real_in_target_mask, :][:, real_in_target_mask]
            loss_fea2mp_list.append(bceloss(pred_adj_real, label_adj_real))
        loss_fea2mp_test = sum(loss_fea2mp_list)
        loss_GNN_test = F.nll_loss(out[test_idx], labels[test_idx])
        loss_test = loss_GNN_test + args.alpha * loss_fea2mp_test + args.beta * loss_embed

        _, preds_test = torch.max(out[test_idx].data, dim=1)
        acc_test = torch.sum(preds_test == labels[test_idx]).to(torch.float32) / test_idx.shape[0]

        preds_test_list = preds_test.cpu().numpy().tolist()
        macro_f1 = f1_score(label_test_list, preds_test_list, average='macro')

        prediction_test = out[test_idx].cpu().numpy().tolist()
        roc = roc_auc_score(labels_test_one_hot, prediction_test, multi_class='ovo')

    time_end_test = time.time()
    res.append('Test_loss:{:.4f} | Test_acc:{:.4f} | Test_macro_f1:{:.4f} | Test_ROC:{:.4f}'
               .format(loss_test.item(), acc_test, macro_f1, roc))

    res_acc[i] = acc_test
    res_micro[i] = micro_f1
    res_macro[i] = macro_f1
    res_roc[i] = roc
    res_train_time[i] = (time_end_train - time_start_train) / (epoch + 1)
    res_test_time[i] = (time_end_test - time_start_test)

print("===========================")
print("acc: ", process_result(res_acc))
print("micro_f1: ", process_result(res_micro))
print("macro_f1: ", process_result(res_macro))
print("ROC: ", process_result(res_roc))
