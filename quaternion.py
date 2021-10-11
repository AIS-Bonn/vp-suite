import math

import torch
import numpy as np

"""
sources: 
- https://github.com/facebookresearch/QuaterNet/blob/main/common/quaternion.py
- pyquaternion
"""

def q_mul(q1, q2):
    """
    Multiply quaternion(s) q with quaternion(s) r.
    Expects two equally-sized tensors of shape (*, 4), where * denotes any number of dimensions.
    Returns q1*q2 as a tensor of shape (*, 4).
    """
    assert q1.shape[-1] == 4
    assert q2.shape[-1] == 4

    original_shape = q1.shape

    # Compute outer product
    terms = torch.bmm(q2.view(-1, 4, 1), q1.view(-1, 1, 4))
    w = terms[:, 0, 0] - terms[:, 1, 1] - terms[:, 2, 2] - terms[:, 3, 3]
    x = terms[:, 0, 1] + terms[:, 1, 0] - terms[:, 2, 3] + terms[:, 3, 2]
    y = terms[:, 0, 2] + terms[:, 1, 3] + terms[:, 2, 0] - terms[:, 3, 1]
    z = terms[:, 0, 3] - terms[:, 1, 2] + terms[:, 2, 1] + terms[:, 3, 0]

    return torch.stack((w, x, y, z), dim=1).view(original_shape)


def wrap_angle(theta):
    """Helper method: Wrap any angle to lie between -pi and pi
    Odd multiples of pi are wrapped to +pi (as opposed to -pi)
    """
    pi = torch.ones_like(theta, device=theta.device) * math.pi
    result = ((theta + pi) % (2 * pi)) - pi
    result[result.eq(-pi)] = pi

    return result


def q_angle(q):

    assert q.shape[-1] == 4

    q = q_normalize(q)
    q_re, q_im = torch.split(q, [1, 3], dim=-1)
    norm = torch.sqrt(torch.sum(torch.square(q_im), dim=-1))
    angle = 2.0 * torch.atan2(norm, q_re)

    return wrap_angle(angle)


def q_normalize(q):

    assert q.shape[-1] == 4

    norm = torch.sqrt(torch.sum(torch.square(q), dim=-1))  # ||q|| = sqrt(w²+x²+y²+z²)
    assert not torch.any(torch.isclose(norm, torch.zeros_like(norm, device=q.device)))  # check for singularities
    return  torch.div(q, norm[:, None])  # q_norm = q / ||q||

def q_conjugate(q)

    assert q.shape[-1] == 4

    conj = torch.tensor([1, -1, -1, -1], device=q.device)  # multiplication coefficients per element
    return q * conj.expand_as(q)


def dq_mul(dq1, dq2):

    assert dq1.shape[-1] == 8
    assert dq2.shape[-1] == 8

    dq1_r, dq1_d = torch.split(dq1, [4, 4], dim=-1)
    dq2_r, dq2_d = torch.split(dq2, [4, 4], dim=-1)

    dq_prod_r = q_mul(dq1_r, dq2_r)
    dq_prod_d = q_mul(dq1_r * dq2_d) + q_mul(dq1_d, dq2_r)
    dq_prod = torch.cat(dq_prod_r, dq_prod_d, dim=-1)

    return dq_prod


def dq_translation(dq):

    assert dq.shape[-1] == 8

    dq_r, dq_d = torch.split(dq, [4, 4], dim=-1)
    mult = q_mul((2.0 * dq_d), q_conjugate(dq_r))
    return mult[..., 1:]


def dq_normalize(dq):

    assert dq.shape[-1] == 8

    dq_r = dq[..., :4]
    norm = torch.sqrt(torch.sum(torch.square(dq_r), dim=-1))  # ||q|| = sqrt(w²+x²+y²+z²)
    assert not torch.any(torch.isclose(norm, torch.zeros_like(norm, device=dq.device)))  # check for singularities
    return torch.div(dq, norm[:, None])  # dq_norm = dq / ||q|| = dq_r / ||dq_r|| + dq_d / ||dq_r||


def dq_quaternion_conjugate(dq):

    assert dq.shape[-1] == 8

    conj = torch.tensor([1, -1, -1, -1,   1, -1, -1, -1], device=dq.device)  # multiplication coefficients per element
    return dq * conj.expand_as(dq)


def dq_to_screw(dq):

    assert dq.shape[-1] == 8

    dq_r, dq_d = torch.split(dq, [4, 4], dim=-1)
    theta = q_angle(dq_r)
    no_rot = torch.isclose(theta, torch.zeros_like(theta, device=dq.device))
    with_rot = ~no_rot
    dq_t = dq_translation(dq)

    l = torch.zeros(*dq.shape[:-1], 3, device=dq.device)
    m = torch.ones(*dq.shape[:-1], 3, device=dq.device)
    theta = torch.zeros(*dq.shape[:-1], device=dq.device)
    d = torch.zeros(*dq.shape[:-1], device=dq.device)

    l[with_rot] = dq_r[with_rot, 1:] / np.sin(theta / 2)
    d[with_rot] = (dq_t[with_rot] * l[with_rot]).sum(dim=-1)  # batched dot product
    t_l_cross = torch.cross(dq_t[with_rot], l[with_rot], dim=-1)
    m[with_rot] = 0.5 * (t_l_cross + torch.cross(l[with_rot], t_l_cross / np.tan(theta / 2), dim=-1))

    d[no_rot] = torch.linalg.norm(dq_t[no_rot], dim=-1)
    no_trans = torch.isclose(d[no_rot], torch.zeros_like(d[no_rot], device=dq.device))
    unit_transform = torch.logical_and(no_rot, no_trans)
    only_trans = ~unit_transform
    l[unit_transform] = dq_t[unit_transform] / d[unit_transform]
    l[only_trans] = 0
    m[no_rot] *= float("inf")

    return l, m, theta, d



