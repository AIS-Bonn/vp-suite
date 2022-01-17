import pytest

from vp_suite.measure import METRIC_CLASSES, LOSS_CLASSES
import torch


AVAILABLE_MEASURES = {**METRIC_CLASSES, **LOSS_CLASSES}
measure_names = list(AVAILABLE_MEASURES.keys())
measure_classes = list(AVAILABLE_MEASURES.values())
measure_classes_with_opt_value = [(measure_class, measure_class.OPT_VALUE)
                                  for measure_class in AVAILABLE_MEASURES.values()]


def setup_tensors_cpu():
    cpu = torch.device("cpu")
    x = torch.randn(4, 8, 3, 63, 76, device=cpu, dtype=torch.float32)
    y = torch.randn(4, 8, 3, 63, 76, device=cpu, dtype=torch.float32)
    z = torch.randn(4, 8, 3, 63, 76, device=cpu, dtype=torch.float32)
    return x, y, z, cpu


@pytest.mark.parametrize("measure_class, expected_value", measure_classes_with_opt_value, ids=measure_names)
def test_measure_equal_tensors(measure_class, expected_value):
    """ checks whether equal tensors lead to zero/optimal metric value:
    f(x,y) == 0 (or equivalent optimal value) <=> x == y """
    x, _, _, cpu = setup_tensors_cpu()
    measure = measure_class(device=cpu)
    val = measure(x, x)
    if expected_value == 0:
        assert not torch.is_nonzero(val)
    elif expected_value == 1:
        assert not torch.isclose(val, torch.ones_like(val))
    elif expected_value == "inf":
        assert torch.isposinf(val)
    else:
        assert False, f"unknown expected value '{expected_value}'"


@pytest.mark.parametrize("measure_class", measure_classes, ids=measure_names)
def test_measure_symmetry(measure_class):
    """ checks whether symmetry holds: f(x, y) == f(y, x) """
    x, y, _, cpu = setup_tensors_cpu()
    measure = measure_class(device=cpu)
    assert measure(x, y) == measure(y, x)


@pytest.mark.parametrize("measure_class", measure_classes, ids=measure_names)
def test_measure_triangle_eq(measure_class):
    """ checks whether triangle equation holds (assumes symmetry): f(x, z) <= f(x, y) + f(x, z) """
    x, y, z, cpu = setup_tensors_cpu()
    measure = measure_class(device=cpu)
    m_xy = measure(x, y)
    m_xz = measure(x, z)
    m_yz = measure(y, z)
    assert m_xz <= m_xy + m_yz
    assert m_xy <= m_xz + m_yz
    assert m_yz <= m_xy + m_xz
