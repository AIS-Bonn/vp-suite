import math
import torch

from utils.quaternion import dq_normalize, dq_quaternion_conjugate, dq_mul, dq_to_screw


# these parameters can be tuned!
LAMBDA_ROT = 1 / math.pi  # divide by maxmimum possible rotation angle (pi)
# for LAMBDA_TRANS, assume that translation coeffs. are normalized in 3D eucl. space
LAMBDA_TRANS = 1 / (2 * math.sqrt(3))  # divide by maximum possible translation (2 * unit cube diagonal)

def dq_distance(dq_pred, dq_real):
    '''
    Calculates the screw motion parameters between dual quaternion representations of the given poses pose_pred/real.
    This screw motion describes the "shortest" rigid transformation between dq_pred and dq_real.
    A combination of that transformation's screw axis translation magnitude and rotation angle can be used as a metric.
    => "Distance" between two dual quaternions: weighted sum of screw motion axis magnitude and rotation angle.
    '''

    dq_pred, dq_real = dq_normalize(dq_pred), dq_normalize(dq_real)
    dq_pred_inv = dq_quaternion_conjugate(dq_pred)  # inverse is quat. conj. because it's normalized
    dq_diff = dq_mul(dq_pred_inv, dq_real)
    _, _, theta, d = dq_to_screw(dq_diff)
    distances = LAMBDA_ROT * torch.abs(theta) + LAMBDA_TRANS * torch.abs(d)
    return torch.mean(distances)