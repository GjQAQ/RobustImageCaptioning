import json
import os

from .base import *
from .occlude import *
from .blur import *


__corrupters = {
    'blur': GaussianBlur,
    'preserve': Preserve,
    'occlude': Occlude
}

__dir_path = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(__dir_path, 'defaults.json')) as f:
    __defaults = json.load(f)


def get_corrupter(corrupt_type: str) -> Corrupter:
    """
    Available corrupters include 'blur' and 'preserve'.
    """
    return __corrupters[corrupt_type]


def get_default_params(corrupt_type: str) -> dict:
    return __defaults[corrupt_type]


def get_instance(corrupt_type: str) -> Corrupter:
    return get_corrupter(corrupt_type)(**get_default_params(corrupt_type))
