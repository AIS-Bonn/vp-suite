from quaternion import dq_mul, dq_normalize, dq_quaternion_conjugate, dq_to_screw

LAMBDA_ROT = 1
LAMBDA_TRANS = 1

def dual_quaternion_distance(pose_pred, pose_real):
    '''
    Calculates the screw motion parameters between dual quaternion representations of the given poses pose_pred/real.
    This screw motion describes the "shortest" path between dq_pred and dq_real.
    A combination of the screw axis translation magnitude and the rotation angle can be used as a metric.
    '''

    dq_pred, dq_real = dq_normalize(pose_pred), dq_normalize(pose_real)
    dq_pred_inv = dq_quaternion_conjugate(dq_pred)  # inverse is quat. conj. because it's normalized
    dq_diff = dq_mul(dq_pred_inv, dq_real)
    _, _, theta, d = dq_to_screw(dq_diff)
    return torch.mean(LAMBDA_ROT * theta + LAMBDA_TRANS * d)
