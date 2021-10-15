import math
import torch
from scipy.spatial.transform import Rotation
from dual_quaternions import DualQuaternion
from losses.pose_distance import dq_distance
from utils.quaternion import q_from_re


def identity(obj): return obj


def dq_from_rq_tv(pose_rq_tv):
    """
    TODO doc
    """
    device = pose_rq_tv.device
    pose_rq_tv_list = pose_rq_tv.tolist()
    pose_dq = [DualQuaternion.from_quat_pose_array(rq_tv).dq_array() for rq_tv in pose_rq_tv_list]
    return torch.tensor(pose_dq, device=device, dtype=torch.float)


def dq_from_re_tv(pose_re_tv):
    """
    TODO doc
    """
    device = pose_re_tv.device
    pose_re, pose_tv = torch.split(pose_re_tv, [3, 3], dim=-1)
    pose_rq_list = q_from_re(pose_re).tolist()
    pose_tv_list = pose_tv.tolist()
    pose_dq = [DualQuaternion.from_quat_pose_array(rq + tv).dq_array() for rq, tv in zip(pose_rq_list, pose_tv_list)]
    return torch.tensor(pose_dq, device=device, dtype=torch.float)


def dq_from_tv(pose_tv):
    """
    TODO doc
    """
    device = pose_tv.device
    pose_tv_list = pose_tv.tolist()
    pose_dq = [DualQuaternion.from_quat_pose_array([1, 0, 0, 0] + tv).dq_array() for tv in pose_tv_list]
    return torch.tensor(pose_dq, device=device, dtype=torch.float)


def pose_distance_in_dq_space(pose_pred, pose_real, graph_mode):
    """
    TODO doc
    """
    dq_pred = convert_to_dq[graph_mode](pose_pred)
    dq_real = pose_real  # pose_real should be provided in dual quaternion rep.
    return dq_distance(dq_pred, dq_real)


convert_to_dq = {
    "tv": NotImplemented,
    "re_tv": dq_from_re_tv,
    "rq_tv": dq_from_rq_tv,
    "rq_tv_to_tv": NotImplemented,
    "dq": identity,
}  # even though dq_from_tv() is implemented, for metrics it shall not be used since rotations should have been copied