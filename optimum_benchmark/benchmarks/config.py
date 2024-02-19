from abc import ABC
from dataclasses import dataclass
from logging import getLogger
from typing import TypeVar

LOGGER = getLogger("benchmark")


@dataclass
class BenchmarkConfig(ABC):
    name: str
    _target_: str

    def __post_init__(self):
        pass


BenchmarkConfigT = TypeVar("BenchmarkConfigT", bound=BenchmarkConfig)
