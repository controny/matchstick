from hypothesis import settings
from hypothesis.strategies import floats, integers

from minitorch import operators

settings.register_profile("ci", deadline=None)
settings.load_profile("ci")


small_ints = integers(min_value=1, max_value=3)
small_floats = floats(min_value=-100, max_value=100, allow_nan=False)
med_floats = floats(min_value=101, max_value=300, allow_nan=False)
large_floats = floats(min_value=301, max_value=500, allow_nan=False)
med_ints = integers(min_value=1, max_value=20)


def assert_close(a: float, b: float) -> None:
    assert operators.is_close(a, b), "Failure x=%f y=%f" % (a, b)
