import os


def get_allowed_n_proc():
    return min(16, os.cpu_count())
