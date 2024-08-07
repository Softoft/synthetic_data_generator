import logging
import math
from abc import ABC
from dataclasses import dataclass, field

import numpy as np


@dataclass
class NumberInterval:
    lower_bound: float
    upper_bound: float

    @classmethod
    def get_positive_interval(cls):
        return cls(0, 1e40)

    @classmethod
    def get_infinity_interval(cls):
        return cls(-1e40, 1e40)

    def __contains__(self, item: float | int):
        return self.lower_bound <= item <= self.upper_bound


class NumberGenerator(ABC):
    def generate_bounded_number(self, mean: float, standard_deviation: float, number_bounds: NumberInterval) -> int:
        pass


@dataclass
class NormalizedNumberGenerator(NumberGenerator):
    def _generate_random_normal_distribution_number(self, mean, standard_deviation) -> int:
        return round(np.random.normal(mean, standard_deviation))

    def generate_bounded_number(self, mean: float, standard_deviation: float, number_bounds: NumberInterval) -> int:
        number = self._generate_random_normal_distribution_number(mean, standard_deviation)
        while number not in number_bounds:
            number = self._generate_random_normal_distribution_number(mean, standard_deviation)
        return number


@dataclass
class NumberIntervalGenerator:
    """
    Generates a random number interval based on a normal distribution.
    Attributes:
        mean (float): The mean of the normal distribution.
        standard_deviation (float): The standard deviation of the normal distribution.
        min_upper_bound_difference (float): The minimum difference between the lower and upper bounds.
        lower_number_bounds (NumberInterval): The bounds for the lower number.
        lower_number_generator (NormalizedNumberGenerator): Optional. The generator for the lower number.
    """
    mean: float
    lower_number_generator: NumberGenerator
    standard_deviation: float
    min_upper_bound_difference: float
    lower_number_bounds: NumberInterval = field(
        default_factory=lambda: NumberInterval.get_positive_interval())

    def _generate_upper_bound(self, lower_bound: int) -> int:
        upper_bound_factor = math.log2(max(2, abs(lower_bound)))
        return round(lower_bound + self.min_upper_bound_difference * upper_bound_factor)

    def generate_bounds(self) -> NumberInterval:
        lower_bound = self.lower_number_generator.generate_bounded_number(self.mean, self.standard_deviation,
                                                                          self.lower_number_bounds)
        upper_bound = self._generate_upper_bound(lower_bound)
        logging.info(f"Generated number interval: {lower_bound}, {upper_bound}")
        return NumberInterval(lower_bound=lower_bound, upper_bound=upper_bound)
