import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self, node_feat_dim):
        super().__init__()
        self.node_feat_dim = node_feat_dim
        self.conv1 = GCNConv(self.node_feat_dim, 16)
        self.conv2 = GCNConv(16, 7)  # 7D pose (quaternions for rotation)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        return x