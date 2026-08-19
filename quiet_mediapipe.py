
import os
import sys
import time
from contextlib import contextmanager

import numpy as np


@contextmanager
def suppress_native_stderr():
    stderr_fd = sys.stderr.fileno()
    saved_fd = os.dup(stderr_fd)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull_fd, stderr_fd)
        yield
    finally:

        sys.stderr.flush()
        time.sleep(0.05)
        os.dup2(saved_fd, stderr_fd)
        os.close(devnull_fd)
        os.close(saved_fd)


def create_pose_model(mp_pose, **pose_kwargs):

    with suppress_native_stderr():
        pose_model = mp_pose.Pose(**pose_kwargs)
        pose_model.process(np.zeros((2, 2, 3), dtype=np.uint8))
    return pose_model
