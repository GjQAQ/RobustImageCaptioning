from .base import *
from .occlude import *
from .blur import *


__corrupters = {
    'blur': GaussianBlur,
    'preserve': Preserve
}


def get_corrupter(corrupt_type: str) -> Corrupter:
    return __corrupters[corrupt_type]
