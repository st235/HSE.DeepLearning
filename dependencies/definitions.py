import os
import sys
import torch

_DIRECTORY_CURRENT = os.path.dirname(os.path.abspath(__file__))


def __get_dependency_folder(dependency_name: str) -> str:
    return os.path.join(_DIRECTORY_CURRENT, dependency_name)


def load_module(module: str):
    sys.path.append(__get_dependency_folder(module))


def get_file_path(*paths) -> str:
    return os.path.join(_DIRECTORY_CURRENT, *paths)


def fetch_model(dependency_name: str, model_path: str):
    return torch.hub.load(__get_dependency_folder(dependency_name), 'custom',
                          source='local',
                          trust_repo=True,
                          path=os.path.join(_DIRECTORY_CURRENT, model_path))
