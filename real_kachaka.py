from kachaka import KachakaBase, Pose
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Image, display
import kachaka_api


class RealKachaka(KachakaBase):
    def __init__(self):
        super().__init__()
        self.client = kachaka_api.aio.KachakaApiClient()

    def move_to_pose(self, distination: Pose):
        pass

    def is_moving_finished(self):
        pass

    def update_sensor_data(self):
        pass
