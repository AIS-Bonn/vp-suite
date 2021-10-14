import os, json

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation
from dual_quaternions import DualQuaternion

from torch_geometric_temporal.signal import DynamicGraphTemporalSignal
from pytorch3d.transforms import matrix_to_euler_angles, matrix_to_quaternion
from utils.quaternion import q_to_re

NODE_FEAT_DIM = {
    "tv": (4, 3),
    "re_tv": (7, 6),
    "rq_tv": (8, 7),
    "rq_tv_to_tv": (8, 3),
    "dq": (9, 8)
}  # (in_dim, out_dim)


class SynpickGraphDataset(object):
    def __init__(self, data_dir, num_frames, step, allow_overlap, graph_mode="tv"):
        super(SynpickGraphDataset, self).__init__()

        scene_dir = os.path.join(data_dir, 'scene_gt')
        self.scene_ids = {int(scene_fp[-20:-14]): os.path.join(scene_dir, scene_fp)
                          for scene_fp in sorted(os.listdir(scene_dir))}

        self.skip_first_n = 72
        self.step = step  # if >1, (step - 1) frames are skipped between each frame
        self.sequence_length = (num_frames - 1) * self.step + 1  # num_frames also includes prediction horizon
        self.frame_offsets = range(0, num_frames * self.step, self.step)

        # Frames are packed like [[x_0, x_1, x_2, ...], [x_1, x_2, x_3...], ...] if overlap is allowed.
        # Otherwise, [[x_0, x_1, x_2, ...], [x_3, x_4, x_5, ...], ...]
        self.allow_overlap = allow_overlap
        self.action_size = 3
        self.graph_mode = graph_mode
        self.node_feat_dim = NODE_FEAT_DIM[self.graph_mode]

        # determine which dataset indices are valid for given sequence length T
        self.n_total_frames = 0
        self.valid_frames = []
        for ep, scene_fp in self.scene_ids.items():
            ep = int(scene_fp[-20:-14])
            with open(scene_fp, "r") as scene_json_file:
                scene_dict = json.load(scene_json_file)

            # discard episode if it includes objects that fell out of the bin (-> obj. height exceeds max_height)
            ep_obj_pos = [obj_dict["cam_t_m2c"] for frame_list in scene_dict.values() for obj_dict in frame_list]
            if any([outside_tote(pos) for pos in ep_obj_pos]):
                print(f"skipping episode {ep} because an object is outside the tote")
                continue

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

        if len(self.valid_frames) < 1:
            raise ValueError("No valid indices in generated dataset! "
                             "Perhaps the calculated sequence length is longer than the trajectories of the data?")


    def get_pose(self, r_matvec, t):

        R = torch.tensor([r_matvec[i:i + 3] for i in [0, 3, 6]])
        r = None
        if "q" in self.graph_mode:
            r = matrix_to_quaternion(R).numpy()
        elif "re" in self.graph_mode:
            r = q_to_re(matrix_to_quaternion(R)).numpy()
        pose = t if r is None else list(np.concatenate([r, t]))
        if self.graph_mode == "dq":
            pose = DualQuaternion.from_quat_pose_array(pose).dq_array()
        return pose


    def __getitem__(self, i):

        ep, start_frame = self.valid_frames[i]  # only consider valid indices
        frame_list = self.get_scene_frames(ep, start_frame)  # list of lists of object information dicts

        gripper_pos = [frame_info[-1]["cam_t_m2c"] for frame_info in frame_list]
        gripper_pos = np.array(gripper_pos + gripper_pos[-1:])  # duplicate last pos to pad the resulting actions array
        gripper_actions = [normalize_a(new - old) for old, new in zip(gripper_pos, gripper_pos[1:])]  # actions = gripper t-deltas

        edge_indices, edge_weights, node_features, targets = [], [], [], []
        for frame_info in frame_list:
            all_instance_ids = [obj_info["ins_id"] for obj_info in frame_info]
            edge_indices_t, node_features_t = [], []
            for obj_info in frame_info:

                # assemble node feature vector
                R = obj_info["cam_R_m2c"]
                t_vec = normalize_t(obj_info["cam_t_m2c"])
                pose = self.get_pose(R, t_vec)
                obj_class = np.array([obj_info["obj_id"]])
                obj_feature = np.concatenate([pose, obj_class])  # object's feature (x) vector
                node_features_t.append(obj_feature)

                # assemble outgoing edges
                instance_idx = obj_info["ins_id"]  # object's instance idx
                node_idx = all_instance_ids.index(instance_idx)
                touches = obj_info.get("touches", [idx for idx in all_instance_ids if idx != instance_idx])  # list of instance idx this object touches
                touched_node_idx = [all_instance_ids.index(t) for t in touches]
                edge_indices_t.extend([np.array([node_idx, touch_idx]) for touch_idx in touched_node_idx])

            node_features.append(np.stack(node_features_t, axis=0))  # shape: [|V|, feat_dim]
            edge_indices_t = np.stack(edge_indices_t, axis=1)
            edge_indices.append(edge_indices_t)  # shape: [2, |E|]
            edge_weights.append(np.ones(edge_indices_t.shape[1]))  # shape: [|E|]
            targets.append(None)

        gripper_actions = [np.expand_dims(ga, 0).repeat(nf.shape[0], axis=0)
                           for ga, nf in zip(gripper_actions, node_features)]

        data_signal = DynamicGraphTemporalSignal(
            edge_indices = edge_indices,
            edge_weights = edge_weights,
            features = node_features,
            targets = targets,
            action = gripper_actions
        )  # sequence consists of T graphs

        return data_signal

    def get_dataset(self):
        return self.__getitem__(0)


    def __len__(self):
        return len(self.valid_frames)


    def get_scene_frames(self, ep : int, start_frame : int):
        '''
        This function assumes that all given ep-frame combinations exist!
        '''
        with open(self.scene_ids[ep], "r") as scene_json_file:
            ep_dict = json.load(scene_json_file)
        sequence_frames = [start_frame + offset for offset in self.frame_offsets]
        return [ep_dict[str(frame)] for frame in sequence_frames]


tote_min_coord = [-300, -200, 1500]
tote_max_coord = [ 300,  200, 2000]

def outside_tote(pos):
    return pos[0] < tote_min_coord[0]\
           or pos[1] < tote_min_coord[1]\
           or pos[2] < tote_min_coord[2]\
           or pos[0] > tote_max_coord[0]\
           or pos[1] > tote_max_coord[1]\
           or pos[2] > tote_max_coord[2]

def normalize_t(pos):
    return 2 * np.divide(np.array(pos) - tote_min_coord, np.array(tote_max_coord) - tote_min_coord) - 1

def normalize_a(action):
    return np.divide(action, [20, 20, 20])

def denormalize_t(pos):

    pos_dim = pos.shape[-1]
    max_c = tote_max_coord[0:pos_dim]
    min_c = tote_min_coord[0:pos_dim]
    return ((np.array(pos) + 1) / 2) * (np.array(max_c) - min_c) + min_c