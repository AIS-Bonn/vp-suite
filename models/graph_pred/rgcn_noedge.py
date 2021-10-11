import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from torch_geometric.data import Batch as GraphBatch
from torch_geometric_temporal.nn.recurrent import DCRNN
from torch_geometric_temporal.signal import DynamicGraphTemporalSignal as DGTS

from losses.dq_distance import dq_distance


class NodeToEdge(nn.Module):
    def __init__(self, in_features, create_loops=False):
        super(NodeToEdge, self).__init__()
        self.edge_final = nn.Sequential(
            nn.Linear(2 * in_features, 2 * in_features),
            nn.ReLU(),
            nn.Linear(2 * in_features, 2), # num edge types (incl. "type" non-edge)
            nn.Softmax(dim=-1)
        )

    def forward(self, node_feat, batch_idx, device):

        #  TODO optimize for performance! -> Sparse tensors?

        # create block-diagonal matrix with ones for all node indices that could be linked by an edge
        possible_edge_diag = torch.block_diag(*[torch.ones(v, v, device=device) for v in torch.bincount(batch_idx)])  # [|V|, |V|]
        possible_edge_diag -= torch.eye(possible_edge_diag.shape[0], device=device)  # self-edges get special treatment

        # create concat combinations of all node pairs
        node_feat_L, node_feat_R = node_feat.unsqueeze(1), node_feat.unsqueeze(0)
        node_feat_L = node_feat_L.repeat(1, node_feat_R.shape[1], 1)
        node_feat_R = node_feat_R.repeat(node_feat_L.shape[0], 1, 1)
        all_node_feat_combinations = torch.cat([node_feat_L, node_feat_R], -1) # [|V|, |V|, 2*node_dim]

        # create combinations of all edge indices
        V = node_feat.shape[0]
        node_idx = torch.arange(V, device=device)
        all_edge_index_combinations \
            = torch.stack([node_idx[:, None].repeat(1, V), node_idx[None, :].repeat(V, 1)], dim=-1)  # [|V|, |V|, 2]

        # remove node feature / edge index pairs that should not be
        combined_edge_index = all_edge_index_combinations[possible_edge_diag > 0]  # [|E|', 2]
        combined_node_feat = all_node_feat_combinations[possible_edge_diag > 0]  # [|E|', 2*node_dim]

        # calculate weights of edges and remove edges the model does not propose
        combined_edge_probs = self.edge_final(combined_node_feat)  # [|E|', 2]
        combined_edge_types = Categorical(combined_edge_probs).sample()  # [|E|']
        out_edge_index = combined_edge_index[combined_edge_types > 0]  # [|E|, 2]  # currently, only 1 edge type supported

        # HOTFIX: add self-edges with weight 1 to graph
        self_edges, self_edge_weights = node_idx[:, None].repeat(1, 2), torch.ones(V, device=device)
        out_edge_index = torch.cat([out_edge_index, self_edges], dim=0)

        # TODO variable edge weights?
        out_edge_weights = torch.ones(out_edge_index.shape[0], device=0)  # [|E|]

        return out_edge_index.t(), out_edge_weights


class ObjectPoseEstimator(nn.Module):

    hidden_dim = 32

    def __init__(self, node_features, out_features):
        super(ObjectPoseEstimator, self).__init__()
        self.node_embed = nn.Linear(node_features, self.hidden_dim)
        self.node_rnn = nn.LSTMCell(input_size=self.hidden_dim, hidden_size=self.hidden_dim)
        self.edge_predictor = NodeToEdge(self.hidden_dim)
        self.graph_rnn = DCRNN(self.hidden_dim, self.hidden_dim, 1)
        self.final_linear = nn.Linear(self.hidden_dim, out_features)
        self.loss_mode = "dq" if out_features = 8 else "mse"


    def node_loss(self, pred_x, target_x):
        if self.loss_mode == "dq":
            target_x = target_x[:, :8]
            return dq_distance(pred_x, target_x)
        else:
            target_x = target_x[:, 4:6]
            return F.mse_loss(pred_x, target_x)


    def forward(self, node_in_x, batch_idx, node_rnn_h, node_rnn_c, graph_rnn_h, device):
        node_in_x = self.node_embed(node_in_x)  # [|V|, hidden_dim]
        node_rnn_h = node_rnn_h if node_rnn_h is not None else torch.zeros_like(node_in_x, device=device)
        node_rnn_c = node_rnn_c if node_rnn_c is not None else torch.zeros_like(node_in_x, device=device)
        node_rnn_h, node_rnn_c = self.node_rnn(node_in_x, (node_rnn_h, node_rnn_c))
        edge_index, edge_weight = self.edge_predictor(node_rnn_h, batch_idx, device)
        graph_rnn_h = self.graph_rnn(node_in_x, edge_index, edge_weight, graph_rnn_h)
        node_out_x = self.final_linear(F.relu(graph_rnn_h))
        return node_out_x, edge_index, edge_weight, node_rnn_h, node_rnn_c, graph_rnn_h


    def pred_n(self, input_signal, device, pred_length=1, **kwargs):

        snapshots = [snap for snap in iter(input_signal)]
        T = len(snapshots)
        input_length = T - pred_length
        out_frames = [snapshots[0].to(device)]
        pred_loss = 0
        node_rnn_h, node_rnn_c, graph_rnn_h = None, None, None

        for t in range(T - 1):

            graph_in, graph_target = out_frames[-1], snapshots[t+1].to(device).clone()
            pred_x, pred_edge_index, pred_edge_weight, node_rnn_h, node_rnn_c, graph_rnn_h \
                = self.forward(graph_in.x, graph_in.batch, node_rnn_h, node_rnn_c, graph_rnn_h, device)

            if t >= input_length:  # prediction mode
                pred_loss += self.node_loss(pred_x, graph_target.x)

                # construct Batch object directly because PyG-T does so
                graph_pred = GraphBatch(x=pred_x,
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