import torch
import torch.nn as nn
import torch.nn.functional as F


class fea2mp_head(nn.Module):
    def __init__(self, in_channel, out_sz, dropout):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channel, out_sz, 1, bias=False)
        self.conv2_1 = nn.Conv1d(out_sz, 1, 1)
        self.conv2_2 = nn.Conv1d(out_sz, 1, 1)
        self.dropout = dropout
        self.sigmoid = nn.Sigmoid()

    # 输入x的形状是[ 1, ft_size, target_num ]
    def forward(self, x):
        seq = x.float()
        seq = F.dropout(seq, self.dropout, training=self.training).float()
        seq_fts = self.conv1(seq)
        f_1 = self.conv2_1(seq_fts)
        f_2 = self.conv2_2(seq_fts)

        logits = f_1 + torch.transpose(f_2, 2, 1)  # [1, target_node, target_node ]
        preds = torch.squeeze(logits, dim=0)  # [target_nodes, target_nodes]
        # preds = self.sigmoid(preds)  # [target_node, target_node ]
        return preds
