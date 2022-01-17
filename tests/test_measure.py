import pytest

from vp_suite.measure import METRIC_CLASSES, LOSS_CLASSES
import torch


"""
These tests are prob. useless since they check the metric axioms on single instances
"""


AVAILABLE_MEASURES = {**METRIC_CLASSES, **LOSS_CLASSES}
measure_names = list(AVAILABLE_MEASURES.keys())
measure_classes = list(AVAILABLE_MEASURES.values())


def setup_tensors_cpu():
    cpu = torch.device("cpu")
    x = torch.rand(4, 10, 3, 63, 76, device=cpu, dtype=torch.float32)
    y = torch.rand(4, 10, 3, 63, 76, device=cpu, dtype=torch.float32)
    z = torch.rand(4, 10, 3, 63, 76, device=cpu, dtype=torch.float32)
    return x, y, z, cpu


measure_classes_with_opt_value = [(mc, mc.OPT_VALUE) for mc in AVAILABLE_MEASURES.values()]
@pytest.mark.parametrize("measure_class, expected_value", measure_classes_with_opt_value, ids=measure_names)
def test_measure_equal_tensors(measure_class, expected_value):
    """ checks whether equal tensors lead to zero/optimal metric value:
    f(x,y) == 0 (or equivalent optimal value) <=> x == y """
    x, _, _, cpu = setup_tensors_cpu()
    measure = measure_class(device=cpu)
    val = measure.to_display(measure(x, x))
    if expected_value == 0:
        assert torch.abs(val) < 1e-4
    elif expected_value == 1:
        assert torch.abs(val - 1) < 1e-4
    elif expected_value == float('inf'):
        assert val > 50
    elif expected_value == float('-inf'):
        assert val < -50
    else:
        assert False, f"unknown expected value '{expected_value}'"


@pytest.mark.parametrize("measure_class", measure_classes, ids=measure_names)
def test_measure_symmetry(measure_class):
    """ checks whether symmetry holds: f(x, y) == f(y, x) """
    x, y, _, cpu = setup_tensors_cpu()
    measure = measure_class(device=cpu)
    assert torch.abs(measure(x, y) - measure(y, x)) < 1e-4


measures_triangle_eq = [(mn, mc) for (mn, mc) in AVAILABLE_MEASURES.items()
                             if not mc.BIGGER_IS_BETTER and mc.OPT_VALUE == 0]
measure_names_triangle_eq = map(lambda x: x[0], measures_triangle_eq)
measure_classes_triangle_eq = map(lambda x: x[1], measures_triangle_eq)
@pytest.mark.parametrize("measure_class", measure_classes_triangle_eq, ids=measure_names_triangle_eq)
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
