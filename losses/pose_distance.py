import math
import torch

from utils.quaternion import *


# these parameters can be tuned!
LAMBDA_ROT = 1 / math.pi  # divide by maxmimum possible rotation angle (pi)
# for LAMBDA_TRANS, assume that translation coeffs. are normalized in 3D eucl. space
LAMBDA_TRANS = 1 / (2 * math.sqrt(3))  # divide by maximum possible translation (2 * unit cube diagonal)


def dq_distance(pose_pred, pose_real):
    '''
    Calculates the screw motion parameters between dual quaternion representations of the given poses pose_pred/real.
    This screw motion describes the "shortest" rigid transformation between dq_pred and dq_real.
    A combination of that transformation's screw axis translation magnitude and rotation angle can be used as a metric.
    => "Distance" between two dual quaternions: weighted sum of screw motion axis magnitude and rotation angle.
    '''
    dq_pred, dq_real = dq_normalize(pose_pred), dq_normalize(pose_real)
    dq_pred_inv = dq_quaternion_conjugate(dq_pred)  # inverse is quat. conj. because it's normalized
    dq_diff = dq_mul(dq_pred_inv, dq_real)
    _, _, theta, d = dq_to_screw(dq_diff)
    distances = LAMBDA_ROT * torch.abs(theta) + LAMBDA_TRANS * torch.abs(d)
    return torch.mean(distances)


def rq_tv_distance(pose_pred, pose_real):
    '''
    TODO doc
    '''
    rq_pred, tv_pred = torch.split(pose_pred, [4, 3], dim=-1)
    rq_real, tv_real = torch.split(pose_real, [4, 3], dim=-1)

    rq_real = q_normalize(rq_real)
    rq_pred_inv = q_conjugate(q_normalize(rq_pred))
    rq_diff = q_mul(rq_pred_inv, rq_real)
    theta = torch.abs(q_angle(rq_diff))

    d = torch.linalg.norm(tv_real - tv_pred, dim=-1)

    distances = LAMBDA_ROT * theta + LAMBDA_TRANS * d
    return torch.mean(distances)


def re_tv_distance(pose_pred, pose_real):
    """
    TODO Doc
    """
    re_pred, tv_pred = torch.split(pose_pred, [3, 3], dim=-1)
    re_real, tv_real = torch.split(pose_real, [3, 3], dim=-1)
    rq_tv_pred = torch.cat([q_from_re(re_pred), tv_pred], dim=-1)
    rq_tv_real = torch.cat([q_from_re(re_real), tv_real], dim=-1)

    return rq_tv_distance(rq_tv_pred, rq_tv_real)


def tv_distance(pose_pred, pose_real):
    '''
    TODO doc
    '''
    d = torch.linalg.norm(pose_pred - pose_real, dim=-1)
    return torch.mean(d)


node_distance_fn = {
    "tv": tv_distance,
    "re_tv": re_tv_distance,
    "rq_tv": rq_tv_distance,
    "rq_tv_to_tv": tv_distance,
    "dq": dq_distance
}