from itertools import permutations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data as GraphData, Batch as GraphBatch
from torch_geometric_temporal.nn.recurrent import DCRNN
from torch_geometric_temporal.signal import DynamicGraphTemporalSignal as DGTS


class NodeToEdge(nn.Module):
    def __init__(self, in_features, create_loops=False):
        super(NodeToEdge, self).__init__()
        self.edge_final = nn.Sequential(
            nn.Linear(2 * in_features, 2 * in_features),
            nn.Linear(2 * in_features, 1),
            nn.ReLU() # capped to 1?
        )

    def forward(self, node_feat, batch_idx, device):

        #  TODO optimize for performance! -> Remove for-loop and list?
        sliced_node_x = [node_feat[batch_idx == i] for i in batch_idx.unique()]
        combined_node_idx, combined_node_x = [], []
        node_start_idx = 0

        for node_x in sliced_node_x:
            node_count = node_x.shape[0]
            node_idx = torch.arange(node_start_idx, node_start_idx + node_count, device=device).unsqueeze(dim=-1)
            comb_node_idx = torch.stack([torch.cat(p, 0) for p in permutations(node_idx, 2)])
            comb_node_x = torch.stack([torch.cat(p, 0) for p in permutations(node_x, 2)])
            combined_node_idx.append(comb_node_idx)
            combined_node_x.append(comb_node_x)
            node_start_idx += node_count

        edge_index = torch.cat(combined_node_idx, 0).t()  # [2, |E|]
        combined_node_x = torch.cat(combined_node_x, 0)  # [|V|, 2*dim]
        edge_weights = self.edge_final(combined_node_x).squeeze(dim=-1)

        return edge_index, edge_weights


class ObjectPoseEstimator(nn.Module):

    hidden_dim = 32

    def __init__(self, node_features, out_features):
        super(ObjectPoseEstimator, self).__init__()
        self.node_embed = nn.Linear(node_features, self.hidden_dim)
        self.edge_predictor = NodeToEdge(self.hidden_dim)
        self.recurrent = DCRNN(self.hidden_dim, self.hidden_dim, 1)
        self.final_linear = nn.Linear(self.hidden_dim, out_features)


    def forward(self, node_x, batch_idx, h, device):
        node_x_embed = self.node_embed(node_x)
        edge_index, edge_weight = self.edge_predictor(node_x_embed, batch_idx, device)
        h = self.recurrent(node_x_embed, edge_index, edge_weight, h)
        out = self.final_linear(F.relu(h))
        return out, h, edge_index, edge_weight


    def pred_n(self, input_signal, device, pred_length=1, **kwargs):

        snapshots = [snap for snap in iter(input_signal)]
        T = len(snapshots)
        input_length = T - pred_length
        out_frames = [snapshots[0].to(device)]
        pred_loss = 0
        rnn_h = None

        for t in range(T - 1):

            graph_in, graph_target = out_frames[-1], snapshots[t+1].to(device).clone()
            pred_x, rnn_h, pred_edge_index, pred_edge_weight \
                = self.forward(graph_in.x, graph_in.batch, rnn_h, device)

            if t >= input_length:  # prediction mode
                pred_loss += F.mse_loss(pred_x, graph_target.x[:, 4:6])

                graph_pred_x = graph_in.x.clone()
                graph_pred_x[:, 4:6] = pred_x

                # construct Batch object directly because PyG-T does so
                graph_pred = GraphBatch(x=graph_pred_x,
                                        edge_index = pred_edge_index,
                                        edge_attr = pred_edge_weight,
                                        y=graph_in.y,
                                        batch=graph_in.batch)
                out_frames.append(graph_pred)
            else:
                out_frames.append(graph_target)

        pred_edges = [snap["edge_index"].detach().cpu().numpy() for snap in out_frames]
        pred_edge_weights = [snap["edge_attr"].detach().cpu().numpy() for snap in out_frames]
        pred_features = [snap["x"].detach().cpu().numpy() for snap in out_frames]

        signal_pred = DGTS(
            edge_indices = pred_edges,
            edge_weights = pred_edge_weights,
            features = pred_features,
            targets = [None] * len(out_frames)
        )  # unusable for losses since tensors are converted to numpy

        pred_loss /= pred_length  # normalize by num. of predicted frames

        return signal_pred, pred_loss