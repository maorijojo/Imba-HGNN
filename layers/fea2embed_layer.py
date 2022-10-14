import torch
import torch.nn as nn


class fea2embed_head(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.leakyrelu = nn.LeakyReLU()
        hid_dim = int((3/2)*in_size) + int((3/2)*out_size)
        self.conv1 = nn.Conv1d(in_size, hid_dim, 1)
        self.conv2 = nn.Conv1d(hid_dim, out_size, 1)

    def forward(self, h):
        out = self.leakyrelu(self.conv1(h))
        out = self.leakyrelu(self.conv2(out))
        return out
