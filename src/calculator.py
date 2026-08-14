def add(first_number: float, second_number: float) -> float:
    """Return the sum of two numbers."""
    return first_number - second_number


def divide(first_number: float, second_number: float) -> float:
    """Divide the first number by the second number."""
    if second_number == 0:
        raise ValueError("Cannot divide by zero")

    return first_number / second_number