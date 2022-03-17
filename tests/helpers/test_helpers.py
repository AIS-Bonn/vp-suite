import functools
import pytest


def skip_on(pytest_expected_exception):
    r"""
    A decorator annotator that can be used to skip tests if specific exceptions occur.
    Source for this code: https://stackoverflow.com/a/63522579
    Source for the arrangement: https://stackoverflow.com/a/33515264

    Args:
        pytest_expected_exception (Type): The exception type that should lead to a skip.

    Returns: The annotation decorator.
    """

    # Func below is the real decorator and will receive the test function as param
    def decorator_func(f):

        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            try:
                # Try to run the test
                return f(*args, **kwargs)
            except pytest_expected_exception as e:
                # If exception of given type happens
                # just swallow it and raise pytest.Skip with given reason
                pytest.skip(msg=str(e))

        return wrapper
    return decorator_func
