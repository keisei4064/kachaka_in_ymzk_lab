from kachaka import KachakaBase, Pose
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Image, display
import asyncio
import kachaka_api


class RealKachaka(KachakaBase):
    def __init__(self):
        super().__init__()
        self.client = kachaka_api.KachakaApiClient()  # 同期ライブラリ
        self.client.update_resolver()
        self.client.set_auto_homing_enabled(False)

    def move_to_pose(self, distination: Pose):
        self.client.move_to_pose(distination.x, distination.y, distination.theta)

    def is_moving_finished(self) -> bool:
        return not self.client.is_command_running()

    def update_sensor_data(self):
        pass
