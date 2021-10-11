import torch
import torch.nn.functional as F
from torch_geometric.data import Data as GraphData, Batch as GraphBatch
from torch_geometric_temporal.nn.recurrent import DCRNN
from torch_geometric_temporal.signal import DynamicGraphTemporalSignal as DGTS


class RecurrentGCN(torch.nn.Module):
    def __init__(self, node_features, out_features):
        super(RecurrentGCN, self).__init__()
        self.recurrent = DCRNN(node_features, 32, 1)
        self.linear = torch.nn.Linear(32, out_features)
        self.loss_mode = "dq" if out_features = 8 else "mse"


    def node_loss(self, pred_x, target_x):
        if self.loss_mode == "dq":
            target_x = target_x[:, :8]
            return dq_distance(pred_x, target_x)
        else:
            target_x = target_x[:, 4:6]
            return F.mse_loss(pred_x, target_x)


    def forward(self, x, edge_index, edge_weight, h=None):
        h = self.recurrent(x, edge_index, edge_weight, h)
        out = self.linear(F.relu(h))
        return out, h


    def pred_n(self, input_signal, device, pred_length=1, **kwargs):

        snapshots = [snap for snap in iter(input_signal)]
        T = len(snapshots)
        input_length = T - pred_length
        out_frames = [snapshots[0].to(device)]
        pred_loss = 0
        rnn_h = None

        for t in range(T - 1):

            graph_in, graph_target = out_frames[-1], snapshots[t+1].to(device).clone()
            pred_x, rnn_h = self.forward(graph_in.x, graph_in.edge_index,
                                         graph_in.edge_attr, rnn_h)

            if t >= input_length:  # prediction mode
                pred_loss += self.node_loss(pred_x, graph_target.x)

                # construct Batch object directly because PyG-T does so
                graph_pred = GraphBatch(x=pred_x,
                                        edge_index = graph_in.edge_index,
                                        edge_attr = graph_in.edge_attr,
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