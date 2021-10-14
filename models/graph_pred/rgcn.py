import random

import torch
import torch.nn.functional as F
from torch_geometric.data import Data as GraphData, Batch as GraphBatch
from torch_geometric_temporal.nn.recurrent import DCRNN
from torch_geometric_temporal.signal import DynamicGraphTemporalSignal as DGTS

from losses.pose_distance import node_distance_fn
from metrics.pose_distance import pose_distance_in_dq_space
from utils.quaternion import check_angle_singularities

class RecurrentGCN(torch.nn.Module):
    def __init__(self, graph_mode, node_in_dim, node_out_dim, include_actions):
        super(RecurrentGCN, self).__init__()
        self.recurrent = DCRNN(node_in_dim, 32, 1)
        self.linear = torch.nn.Linear(32, node_out_dim)
        self.graph_mode = graph_mode
        self.include_actions = include_actions
        self.node_out_dim = node_out_dim


    def set_pose(self, node_x, pose):
        start_ = 4 if self.graph_mode == "rq_tv_to_tv" else 0
        node_x[:, start_:start_+self.node_out_dim] = pose
        return node_x


    def get_pose(self, node_x):
        start_ = 4 if self.graph_mode == "rq_tv_to_tv" else 0
        return node_x[:, start_:start_+self.node_out_dim]


    def node_loss(self, pred_x, target_x):
        return node_distance_fn[self.graph_mode](pred_x, self.get_pose(target_x))


    def node_metric(self, pred_x, target_x):
        return pose_distance_in_dq_space(pred_x, self.get_pose(target_x), self.graph_mode)


    def forward(self, x, edge_index, edge_weight, h=None):
        #print(x.shape)
        h = self.recurrent(x, edge_index, edge_weight, h)
        out = self.linear(F.relu(h))
        return out, h


    def pred_n(self, input_signal, device, pred_length=1, **kwargs):

        teacher_forcing_ratio = kwargs.get("teacher_forcing_ratio", 0)  # non-zero only if specified and during training
        test_mode = kwargs.get("test", False)  # True only during testing -> instead of loss, metric is returned.
        snapshots = [snap for snap in iter(input_signal)]
        T = len(snapshots)
        input_length = T - pred_length
        out_frames = [snapshots[0].to(device)]
        pred_distances = 0
        rnn_h = None

        for t in range(T - 1):

            graph_target = snapshots[t+1].to(device)
            if t < input_length or random.random() < teacher_forcing_ratio:  # obs. phase
                graph_in = snapshots[t].to(device)
            else:  # pred. phase
                graph_in = out_frames[-1]

            in_x = graph_in.x.clone()
            if self.include_actions:  # append action to node features
                in_x = torch.cat([in_x, graph_in.action], dim=-1)
            pred_x, rnn_h = self.forward(in_x, graph_in.edge_index, graph_in.edge_attr, rnn_h)
            if self.graph_mode == "re_tv":
                pred_x = check_angle_singularities(pred_x)
            if test_mode:
                pred_distances += self.node_metric(pred_x, graph_target.x)
            else:
                pred_distances += self.node_loss(pred_x, graph_target.x)

            if t >= input_length:
                # Construct predicted Batch object directly because PyG-T does so. Apart from the nodes' x values,
                # All other attributes are taken from the target graph.
                graph_pred_x = self.set_pose(graph_in.x.clone(), pred_x)
                graph_pred = GraphBatch(x=graph_pred_x,
                                        edge_index = graph_target.edge_index,
                                        edge_attr = graph_target.edge_attr,
                                        y=graph_target.y,
                                        batch=graph_target.batch,
                                        action=graph_target.action)
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

        pred_distances /= pred_length  # normalize by num. of predicted frames

        return signal_pred, pred_distances