import os


def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path
