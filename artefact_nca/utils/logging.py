import os
from functools import wraps

from torch.utils.tensorboard import SummaryWriter


def safe(f):
    """wrap function with try / catch
    Args:
        f ([type]): function to wrap
    """

    @wraps(f)
    def new_func(*args, **kwargs):
        try:
            f(*args, **kwargs)
        except Exception as e:
            print("{} failed with {}".format(f.__name__, e))

    return new_func


class TensorboardLogger:
    def __init__(self, tensorboard_log_path: str = "./tensorboard_logs"):
        self.tensorboard_run_path = os.path.abspath(
            os.path.join(tensorboard_log_path, "default")
        )
        self.tensorboard_writer = self.setup_tensorboard(self.tensorboard_run_path)

    def setup_tensorboard(self, tensorboard_run_path: str):
        tensorboard_writer = SummaryWriter(tensorboard_run_path)
        return tensorboard_writer

    @safe
    def log_scalar(self, data, name="", step=0) -> None:
        if self.tensorboard_writer is not None:
            self.tensorboard_writer.add_scalar(name, data, step)

    @safe
    def log_text(self, data, name="", step=0, description="") -> None:
        if self.tensorboard_writer is not None:
            self.tensorboard_writer.add_text(tag=name, text_string=data)

    @safe
    def close(self):
        self.tensorboard_writer.close()

def doublewrap(f):
    """
    from https://stackoverflow.com/questions/653368/how-to-create-a-python-decorator-that-can-be-used-either-with-or-without-paramet
    answer by @bj0
    a decorator decorator, allowing the decorator to be used as:
    @decorator(with, arguments, and=kwargs)
    or
    @decorator
    """

    @wraps(f)
    def new_dec(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            # actual decorated function
            return f(args[0])
        else:
            # decorator arguments
            return lambda realf: f(realf, *args, **kwargs)

    return new_dec


@doublewrap
def tensorboard(cls):
    class TensorboardWrapper(cls):
        def __init__(self, *args, **kwargs):
            self.logger = TensorboardLogger()
            self.__name__ = cls.__name__
            super().__init__(*args, **kwargs)

        def __repr__(self):
            return repr(cls)

        def log_scalar(self, *args, **kwargs):
            self.logger.log_scalar(*args, **kwargs)

        def log_text(self, *args, **kwargs):
            self.logger.log_text(*args, **kwargs)
        
        def close(self):
            self.logger.close()

    return TensorboardWrapper
