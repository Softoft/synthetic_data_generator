import asyncio

from tests.conftest import KeyEnum, ValueEnum


def test_execute_random_value_multiple_times(create_random_table_node):
    key_value_weight_dict = {
        KeyEnum.K1: { ValueEnum.V1: 1, ValueEnum.V2: 1, ValueEnum.V3: 1 },
        KeyEnum.K2: { ValueEnum.V1: 1, ValueEnum.V2: 1, ValueEnum.V3: 1 },
    }
    random_table_node = create_random_table_node(KeyEnum.K1, key_value_weight_dict)
    storage = asyncio.run(random_table_node.execute())
    random_value = storage.get(ValueEnum)
    assert random_value in [ValueEnum.V1, ValueEnum.V2, ValueEnum.V3]
    for _ in range(10):
        storage = asyncio.run(random_table_node.execute())
        random_value_2 = storage.get(ValueEnum)
        assert random_value_2 == random_value
