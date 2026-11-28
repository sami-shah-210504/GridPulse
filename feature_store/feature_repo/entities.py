from feast import Entity
from feast.value_type import ValueType

region = Entity(
    name="region",
    join_keys=["region"],
    value_type=ValueType.STRING,
    description="Balancing authority (entity key), e.g. CISO",
)
