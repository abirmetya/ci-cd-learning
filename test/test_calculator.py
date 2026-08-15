import pytest

from src.ci_cd_calculator.calculator import add, divide, multiply


def test_add():
    assert add(2, 3) == 5


def test_divide():
    assert divide(10, 2) == 5


def test_multiply():
    assert multiply(2, 3) == 6


def test_divide_by_zero():
    with pytest.raises(ValueError, match="Cannot divide by zero"):
        divide(10, 0)
