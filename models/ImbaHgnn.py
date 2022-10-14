import sys
sys.path.append('../')
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from layers.fea2mp_layer import fea2mp_head
from layers.gat_layer import Attn_head, SimpleAttLayer, sp_Attn_head
from layers.fea2embed_layer import fea2embed_head


class ImbaHgnn(nn.Module):
    def __init__(self, in_dims, adj, meta_path_list, hid_dim, n_hid, num_classes, dropout, n_heads, target_type_mask,
                 real_mask, target_in_real_mask, real_in_target_mask, meta_path_mask, valid_ratio=None, sparse=True):
        super().__init__()
        self.adj = adj
        self.meta_path_list = meta_path_list
        self.hid_dim = hid_dim
        self.dropout = dropout
        self.n_heads = n_heads
        self.n_hid = n_hid
        self.target_type_mask = target_type_mask
        self.target_in_real_mask = target_in_real_mask
        self.real_in_target_mask = real_in_target_mask
        self.real_mask = real_mask
        self.num_meta_path = len(meta_path_list)
        self.valid_ratio = valid_ratio
        self.elu = nn.ELU()
        self.valid_ratio = valid_ratio
        self.fea2mp_layers = nn.ModuleList([fea2mp_head(in_dims[0], n_hid, dropout) for _ in range(self.num_meta_path)])
        self.fea2embed = fea2embed_head(hid_dim, n_hid)
        self.mse_loss = nn.MSELoss()
        self.meta_path_mask = meta_path_mask
        self.fc_list = nn.ModuleList([nn.Linear(in_dim, hid_dim) for in_dim in in_dims])
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)

        if sparse:
            self.full_attn_layers = nn.ModuleList([sp_Attn_head(hid_dim, n_hid, dropout=dropout)
                                                   for _ in range(n_heads)])
            self.full_attn_out = sp_Attn_head(n_heads * n_hid, n_hid, dropout=dropout)
        else:
            self.full_attn_layers = nn.ModuleList([Attn_head(hid_dim, n_hid, dropout=dropout)
                                                   for _ in range(n_heads)])
            self.full_attn_out = Attn_head(n_heads * n_hid, n_hid, dropout=dropout)

        self.meta_attn_layers = self._make_meta_attn_layers()
        self.meta_attn_out = nn.ModuleList([Attn_head(n_hid*n_heads, n_hid, dropout=dropout)
                                            for _ in range(self.num_meta_path)])

        self.reduce_edge_fc = nn.Linear(self.num_meta_path+1, 1)
        nn.init.xavier_normal_(self.reduce_edge_fc.weight, gain=1.414)

        fc_hid_unit = int((2/3)*n_hid) + int((2/3)*num_classes)
        self.out_fc1 = nn.Linear(n_hid, fc_hid_unit)
        nn.init.xavier_normal_(self.out_fc1.weight, gain=1.414)
        self.out_fc2 = nn.Linear(fc_hid_unit, num_classes)
        nn.init.xavier_normal_(self.out_fc2.weight, gain=1.414)

    def _make_meta_attn_layers(self):
        layers = []
        for i in range(self.num_meta_path):
            inner_layers = nn.ModuleList([Attn_head(self.hid_dim, self.n_hid, dropout=self.dropout)
                                          for _ in range(self.n_heads)])
            layers.append(inner_layers)
        return nn.ModuleList(layers)

    def forward(self, features_list, features_target):
        features_target = torch.transpose(torch.unsqueeze(features_target, 0), 2, 1)
        meta_path_preds = [fea2mp(features_target) for fea2mp in self.fea2mp_layers]

        h = [self.elu(fc(features)) for fc, features in zip(self.fc_list, features_list)]
        if len(features_list) > len(self.fc_list):
            h.append(self.elu(self.fc_list[0](features_list[-1])))
        h = torch.cat(h, 0)  # [all_nodes, hid_dim]
        h = torch.transpose(torch.unsqueeze(h, 0), 2, 1)  # [1, hid_dim, all_nodes]

        # h = F.dropout(h, self.dropout, training=self.training)

        h_target = h[:, :, self.target_type_mask]
        h_real = h[:, :, self.real_mask]

        h_1 = torch.cat([att(h_real, self.adj) for att in self.full_attn_layers], dim=1)  # [1, n_heads*n_hid, real_nodes]
        h_1 = self.full_attn_out(h_1, self.adj)  # [1, n_hid, real_nodes]
        h_1 = h_1[:, :, self.target_in_real_mask]  # [1, n_hid, real_target_nodes]

        h_1_pred = self.fea2embed(h_target)  # [1, n_hid, target_nodes]
        h_1_pred_real = h_1_pred[:, :, self.real_in_target_mask]  # [1, n_hid, real_target_nodes]
        loss_embed = self.mse_loss(h_1_pred_real, h_1)

        h_1_pred_fake = h_1_pred[:, :, ~self.real_in_target_mask]  # [1, n_hid, fake_nodes]
        h_1 = torch.cat((h_1, h_1_pred_fake), dim=2)  # [1, n_hid, target_nodes]

        final_meta_path_list = []
        for real_meta_path, pred_meta_path in zip(self.meta_path_list, meta_path_preds):
            final_meta_path_list.append(torch.where(self.meta_path_mask, pred_meta_path, real_meta_path))

        h_2 = [h_1]
        for i in range(self.num_meta_path):
            support = torch.cat([att(h_target, final_meta_path_list[i], is_soft=True)
                                 for att in self.meta_attn_layers[i]], dim=1)
            support = self.meta_attn_out[i](support, final_meta_path_list[i], is_soft=True)
            if self.valid_ratio is not None:
                support = torch.mul(support, self.valid_ratio[i])
            h_2.append(support)

        multi_embed = torch.cat(h_2, dim=0).permute(2, 1, 0)  # [target_nodes, n_hid, num_meta_path+1]
        final_embed = torch.squeeze(self.elu(self.reduce_edge_fc(multi_embed)))  # [target_nodes, n_hid]

        final_embed = F.dropout(final_embed, self.dropout, training=self.training)
        out = self.elu(self.out_fc1(final_embed))
        out = F.log_softmax(self.out_fc2(out), dim=1)

        return meta_path_preds, out, loss_embed