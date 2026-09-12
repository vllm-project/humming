import math
from collections.abc import Iterator


def ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def round_up(value: int, alignment: int) -> int:
    return ceil_div(value, alignment) * alignment


def powers_of_two_up_to(limit: int) -> Iterator[int]:
    value = 1
    while value <= limit:
        yield value
        value *= 2


def positive_divisors(value: int) -> list[int]:
    divisors = []
    for divisor in range(1, math.isqrt(value) + 1):
        if value % divisor:
            continue
        divisors.append(divisor)
        if divisor * divisor != value:
            divisors.append(value // divisor)
    return sorted(divisors)


def is_pow_of_two(n: int) -> bool:
    return n & (n - 1) == 0
