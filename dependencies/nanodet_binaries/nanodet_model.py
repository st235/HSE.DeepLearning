import os

from enum import Enum


class NanoDetPaths(Enum):
    LegacyM = "legacy_nanodet_m.yml", "legacy_nanodet_m.ckpt"
    PlusM320 = "nanodet_plus_m_320.yml", "nanodet_plus_m_320.ckpt"
    PlusM416 = "nanodet_plus_m_416.yml", "nanodet_plus_m_416.ckpt"

    def __init__(self,
                 config_path: str,
                 model_path: str):
        assert config_path.endswith('.yml')
        assert model_path.endswith('.ckpt')

        self.__config_path = config_path
        self.__model_path = model_path

    @property
    def config_path(self) -> str:
        return os.path.abspath(os.path.join(os.path.dirname(__file__), self.__config_path))

    @property
    def model_path(self) -> str:
        return os.path.abspath(os.path.join(os.path.dirname(__file__), self.__model_path))
