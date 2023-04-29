from __future__ import division, absolute_import, print_function

try:
    import tensorboardX as tb
except ImportError:
    print("tensorboardX is not installed")
    tb = None


def add_summary_value(writer, key, value, iteration):
    if writer:
        writer.add_scalar(key, value, iteration)


class TensorBoardLogger:
    def __init__(self, path):
        if tb:
            self.__logger = tb.SummaryWriter(path)

    def train_log(self, k, v, it):
        self.__log(k, v, it, 'train')

    def val_log(self, k, v, it):
        self.__log(k, v, it, 'validation')

    def __log(self, k, v, it, stage):
        if tb:
            self.__logger.add_scalar(f'{stage}/{k}', v, it)
