from src.synthetic_data_generator.random_generators.random_collection import RandomCollection, RandomCollectionFactory
from src.synthetic_data_generator.random_generators.random_collection_interface import IRandom
from src.synthetic_data_generator.random_generators.random_collection_table import RandomTableBuilder
from tests.conftest import BigEnum, KeyEnum, ValueEnum


def get_random_values_distribution(values: list, weights: list, iterations: int, randomization=None) -> list[float]:
    if randomization is not None:
        random_collection: IRandom = RandomCollection(values, weights, randomization_factor=randomization)
    else:
        random_collection: IRandom = RandomCollection(values, weights)
    random_values = [random_collection.get_random_value() for _ in range(iterations)]
    return [random_values.count(value) / len(random_values) for value in values]


def test_random_collection():
    VALUES = [ValueEnum.V1, ValueEnum.V2, ValueEnum.V3]
    WEIGHTS = [0.1, 0.3, 0.6]

    random_value_distribution = get_random_values_distribution(VALUES, WEIGHTS, 100_000)

    for value_distribution, weight in zip(random_value_distribution, WEIGHTS):
        assert abs(value_distribution - weight) < 0.1


def test_random_collection_gets_randomized():
    VALUES = [BigEnum.B1, BigEnum.B2, BigEnum.B3, BigEnum.B4, BigEnum.B5, BigEnum.B6]
    WEIGHTS = [1 / 6 for _ in VALUES]

    random_distribution = get_random_values_distribution(VALUES, WEIGHTS, 100_000, 1)
    random_distribution_randomized = get_random_values_distribution(VALUES, WEIGHTS, 100_000, 1000)

    distribution_weight_diff = [abs(value_distribution - weight) for value_distribution, weight in
                                zip(random_distribution, WEIGHTS)]

    distribution_weight_diff_randomized = [abs(value_distribution - weight) for value_distribution, weight in
                                           zip(random_distribution_randomized, WEIGHTS)]

    assert sum(distribution_weight_diff) + 0.05 < sum(distribution_weight_diff_randomized)


def test_random_enum_collection():
    random_enum_collection = RandomCollectionFactory().build_from_enum(ValueEnum)
    for _ in range(100):
        assert random_enum_collection.get_random_value() in [ValueEnum.V1, ValueEnum.V2, ValueEnum.V3]


def test_build_table_from_dict():
    value_weight_dict = {
        KeyEnum.K1: { ValueEnum.V1: 0, ValueEnum.V2: 0, ValueEnum.V3: 1 },
        KeyEnum.K2: { ValueEnum.V1: 1, ValueEnum.V2: 0, ValueEnum.V3: 0 }
    }

    random_table = RandomTableBuilder().build_from_dict(KeyEnum, ValueEnum, value_weight_dict)
    assert random_table.get_random_value(KeyEnum.K1) is ValueEnum.V3
    assert random_table.get_random_value(KeyEnum.K2) is ValueEnum.V1
