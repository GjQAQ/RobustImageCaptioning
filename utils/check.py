__all__ = ['_check_positive']


def _check_positive(t):
    if not (isinstance(t, (float, int)) and t > 0):
        raise ValueError(f'Wrong temperature value: {t}')
