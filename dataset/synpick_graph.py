import os, json

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R

import networkx as nx
import matplotlib.pyplot as plt

from torch_geometric.data import Data as GraphData
from torch_geometric.data import Dataset as GraphDataset
from torch_geometric.utils.convert import to_networkx

class SynpickGraphDataset(GraphDataset):
    def __init__(self, data_dir, num_frames, step, allow_overlap):
        super(SynpickGraphDataset, self).__init__()

        scene_dir = os.path.join(data_dir, 'scene_gt')
        self.scene_fps = [os.path.join(scene_dir, scene_fp) for scene_fp in sorted(os.listdir(scene_dir))]

        self.skip_first_n = 72
        self.step = step  # if >1, (step - 1) frames are skipped between each frame
        self.sequence_length = (num_frames - 1) * self.step + 1  # num_frames also includes prediction horizon
        self.frame_offsets = range(0, num_frames * self.step, self.step)

        # Frames are packed like [[x_0, x_1, x_2, ...], [x_1, x_2, x_3...], ...] if overlap is allowed.
        # Otherwise, [[x_0, x_1, x_2, ...], [x_3, x_4, x_5, ...], ...]
        self.allow_overlap = allow_overlap
        self.action_size = 3

        # determine which dataset indices are valid for given sequence length T
        self.n_total_frames = 0
        self.valid_frames = []
        for scene_fp in self.scene_fps:

            ep = int(scene_fp[-20:-14])
            with open(scene_fp, "r") as scene_json_file:
                scene_dict = json.load(scene_json_file)
            frames = sorted([int(frame_str) for frame_str in scene_dict.keys()])
            self.n_total_frames += len(frames)
            next_available_idx = 0

            for i in range(self.skip_first_n, len(frames)):

                start_frame = frames[i]
                # discard start_frame if starting from current start_frame would cause overlap
                if start_frame < next_available_idx:
                    continue

                sequence_frames = [start_frame + offset for offset in self.frame_offsets]
                # discard episode if sequences_frames reaches end of episode
                if sequence_frames[-1] > frames[-1]:
                    break

                # all conditions are met -> add start_frame to list of valid sequence_dps and move on
                self.valid_frames.append((ep, start_frame))
                if self.allow_overlap:
                    next_available_idx = start_frame + 1
                else:
                    next_available_idx = start_frame + self.sequence_length

        #print(self.valid_frames)

        if len(self.valid_frames) < 1:
            raise ValueError("No valid indices in generated dataset! "
                             "Perhaps the calculated sequence length is longer than the trajectories of the data?")

    def __getitem__(self, i):

        ep, start_frame = self.valid_frames[i]  # only consider valid indices
        frame_list = self.get_scene_frames(ep, start_frame)  # list of lists of object information dicts

        gripper_pos = np.array([frame_info[-1]["cam_t_m2c"] for frame_info in frame_list])
        gripper_pos_deltas = np.stack([new - old for old, new in zip(gripper_pos, gripper_pos[1:])], axis=0)
        gripper_actions = F.pad(torch.from_numpy(gripper_pos_deltas), (0, 0, 0, 1))  # last graph gets padded action

        # one graph per frame
        seq_graph_data = []
        for frame_info, action in zip(frame_list, gripper_actions):
            all_instance_ids = [obj_info["ins_id"] for obj_info in frame_info]
            node_features = []
            node_pos = []
            node_classes = []
            edges = []
            for obj_info in frame_info:

                # assemble node feature vector
                rotmat = obj_info["cam_R_m2c"]
                r_quat = torch.from_numpy(R.from_matrix([rotmat[i:i+3] for i in [0, 3, 6]]).as_quat())  # rotation in quaternions
                t_vec = torch.tensor(obj_info["cam_t_m2c"])
                class_id = torch.tensor([obj_info["obj_id"]])  # object's class id in tensor shape
                obj_feature = torch.cat([r_quat, t_vec, class_id])  # object's feature vector
                node_features.append(obj_feature)

                # additional features
                node_pos.append(torch.tensor(obj_info["cam_t_m2c"][:2]))
                node_classes.append(class_id)

                # assemble outgoing edges
                instance_idx = obj_info["ins_id"]  # object's instance idx
                node_idx = all_instance_ids.index(instance_idx)
                touches = obj_info.get("touches", [idx for idx in all_instance_ids if idx != instance_idx])  # list of instance idx this object touches
                touched_node_idx = [all_instance_ids.index(t) for t in touches]
                edges.extend([torch.LongTensor([node_idx, touch_idx]) for touch_idx in touched_node_idx])

            # assemble and append graph G = (V, E)
            data = GraphData(x = torch.stack(node_features, dim=0),  # shape: [|V|, feat_dim]
                             edge_index = torch.stack(edges, dim=1),  # shape: [2, |E|]
                             pos = torch.stack(node_pos, dim=0),  # [|V|, 2]
                             obj_class = torch.stack(node_classes, dim=0),  # [|V|, 1]
                             action = action)
            seq_graph_data.append(data)

        return seq_graph_data

    # TODO heterogeneous graph with node_type = class_id?

    def __len__(self):
        return len(self.valid_frames)


    def get_scene_frames(self, ep : int, start_frame : int):
        '''
        This function assumes that all given ep-frame combinations exist!
        '''
        with open(self.scene_fps[ep], "r") as scene_json_file:
            ep_dict = json.load(scene_json_file)
        sequence_frames = [start_frame + offset for offset in self.frame_offsets]
        return [ep_dict[str(frame)] for frame in sequence_frames]


def draw_synpick_graph(graph_data : GraphData, out_fp : str):
    fig = plt.figure(1, figsize=(16, 9))
    G = to_networkx(graph_data, to_undirected=True, node_attrs=["pos", "obj_class"])
    num_nodes = len(G.nodes)
    # add invisible border nodes so that, over time, unchanging obj. positions result in unchanging vis. positions
    G.add_node(num_nodes, pos=(-320, -180), obj_class=24)
    G.add_node(num_nodes+1, pos=(320, -180), obj_class=24)
    G.add_node(num_nodes+2, pos=(-320, 180), obj_class=24)
    G.add_node(num_nodes+3, pos=(320, 180), obj_class=24)

    # color range for object classes
    colors = [G.nodes[i]["obj_class"] / 24 for i in range(num_nodes+4)]
    nx.draw(G, pos=nx.get_node_attributes(G, "pos"), cmap=plt.get_cmap("gist_ncar"),
            node_size=500, linewidths=1, node_color=colors, with_labels=False)
    plt.savefig(out_fp)
    plt.clf()


if __name__ == '__main__':
    import sys, os
    data_dir, out_dir = sys.argv[1], sys.argv[2]
    test_dir = os.path.join(data_dir, 'test')
    test_data = SynpickGraphDataset(data_dir=test_dir, num_frames=16, step=2, allow_overlap=False)

    data = test_data[0]
    for i, frame_graph in enumerate(data):
        draw_synpick_graph(frame_graph, os.path.join(out_dir, f"graph_{i}.png"))