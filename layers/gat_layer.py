import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import spmm
import torch_sparse as torchsp
from torch_scatter import scatter_add, scatter_max
import torch.sparse as sparse


class Attn_head(nn.Module):
    def __init__(self, in_channel, out_sz, dropout, activation=nn.ELU()):
        super().__init__()
        self.dropout = dropout
        self.conv1 = nn.Conv1d(in_channel, out_sz, 1, bias=False)
        self.conv2_1 = nn.Conv1d(out_sz, 1, 1)
        self.conv2_2 = nn.Conv1d(out_sz, 1, 1)
        self.leakyrelu = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=1)
        self.activation = activation
        self.bias = nn.Parameter(torch.Tensor(out_sz), requires_grad=True)
        nn.init.zeros_(self.bias)
        self.zero_val = torch.tensor([-1e9]).float().to(torch.device('cuda'))

    # x: [ 1, ft_size, node_num ]
    def forward(self, x: torch.Tensor, adj: torch.Tensor, is_soft=False):
        seq = x.float()
        seq = F.dropout(seq, self.dropout, training=self.training).float()
        seq_fts = self.conv1(seq)

        f_1 = self.conv2_1(seq_fts)  # 将每个特征结点8维的特征，降低到1维
        f_2 = self.conv2_2(seq_fts)

        logits = f_1 + torch.transpose(f_2, 2, 1)

        if is_soft:
            logits = torch.where(adj > 0, logits*adj, self.zero_val)
        else:
            logits = torch.where(adj > 0, logits, self.zero_val)

        logits = self.leakyrelu(logits)
        coefs = self.softmax(logits)

        coefs = F.dropout(coefs, self.dropout, training=self.training)
        seq_fts = F.dropout(seq_fts, self.dropout, training=self.training)

        ret = torch.matmul(coefs, torch.transpose(seq_fts, 2, 1))  # 对所有结点聚合其邻居结点的特征

        ret = ret + self.bias

        ret = torch.transpose(ret, 2, 1)  # [1, out_sz, num_node]
        return self.activation(ret)


class sp_Attn_head(nn.Module):
    def __init__(self, input_dim, out_dim, dropout, alpha=0.01):
        super().__init__()
        self.in_features = input_dim
        self.out_features = out_dim
        self.alpha = alpha
        self.dropout = dropout
        self.fc = nn.Linear(input_dim, out_dim, bias=False)
        self.fc1 = nn.Linear(out_dim, 1)
        self.fc2 = nn.Linear(out_dim, 1)
        self.reset_parameters()
        # nn.init.xavier_normal_(self.fc.weight, gain=1.414)
        # nn.init.xavier_normal_(self.fc1.weight, gain=1.414)
        # nn.init.xavier_normal_(self.fc2.weight, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.bias = nn.Parameter(torch.Tensor(out_dim), requires_grad=True)
        nn.init.zeros_(self.bias)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.fc.weight.shape[1])
        self.fc.weight.data.uniform_(-stdv, stdv)
        if self.fc.bias is not None:
            self.fc.bias.data.uniform_(-stdv, stdv)

        stdv = 1. / math.sqrt(self.fc1.weight.shape[1])
        self.fc1.weight.data.uniform_(-stdv, stdv)
        if self.fc1.bias is not None:
            self.fc1.bias.data.uniform_(-stdv, stdv)

        stdv = 1. / math.sqrt(self.fc2.weight.shape[1])
        self.fc2.weight.data.uniform_(-stdv, stdv)
        if self.fc2.bias is not None:
            self.fc2.bias.data.uniform_(-stdv, stdv)



    def forward(self, x, adj):
        '''
        :param x:   dense tensor. [1, ft_size, num_nodes]
        :param adj:    parse tensor. size: nodes*nodes
        :return:  hidden features
        '''

        x = torch.squeeze(torch.transpose(x, 2, 1), dim=0)  # [1, ft_size, num_nodes] -> [num_nodes, ft_size]
        x = F.dropout(x, self.dropout, training=self.training)
        N = x.size()[0]   # 图中节点数
        edge = adj._indices()   # 稀疏矩阵的数据结构是indices,values，分别存放非0部分的索引和值，edge则是索引。edge是一个[2, NoneZero]的张量，NoneZero表示非零元素的个数
        h = self.fc(x)  # [num_nodes, hid_dim]
        h_st, h_ed = h[edge[0, :], :], h[edge[1, :], :]  # [num_edge, hid_dim]
        att_st, att_ed = self.fc1(h_st), self.fc2(h_ed)  # [num_edge, 1]
        att_val = att_st + att_ed  # [num_edge, 1]
        att_val = att_val.squeeze()  # [num_edge]
        att_val = self.leakyrelu(att_val)

        att_mx = torch.sparse_coo_tensor(edge, att_val, (N, N), requires_grad=True).cuda()
        coefs = torch.sparse.softmax(att_mx, dim=1)

        h = F.dropout(h, self.dropout, training=self.training)

        _indices, _values = coefs._indices(), coefs._values()
        _values = F.dropout(_values, self.dropout, training=self.training)
        coefs = torch.sparse_coo_tensor(_indices, _values, (N, N), requires_grad=True).cuda()

        ret = torch.sparse.mm(coefs, h)
        ret = ret + self.bias
        ret = torch.transpose(torch.unsqueeze(ret, 0), 2, 1)

        return F.elu(ret)


class SimpleAttLayer(nn.Module):
    def __init__(self, inputs, attention_size=128):
        super(SimpleAttLayer, self).__init__()
        self.hidden_size = inputs
        self.w_omega = nn.Parameter(torch.Tensor(self.hidden_size, attention_size))
        self.b_omega = nn.Parameter(torch.Tensor(attention_size))
        self.u_omega = nn.Parameter(torch.Tensor(attention_size, 1))
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.w_omega)
        nn.init.zeros_(self.b_omega)
        nn.init.xavier_uniform_(self.u_omega)

    # 输入x: [node, num_metapaths, ft_size]
    def forward(self, x):
        v = self.tanh(torch.matmul(x, self.w_omega) + self.b_omega)
        vu = torch.matmul(v, self.u_omega)
        alphas = self.softmax(vu)
        output = torch.sum(x * alphas, dim=1)
        return output