from kachaka import *
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Image, display
import asyncio
import kachaka_api
import pickle


class RealKachaka(KachakaBase):
    def __init__(self):
        super().__init__()
        self.client = kachaka_api.KachakaApiClient()  # 同期ライブラリ
        self.client.update_resolver()
        self.client.set_auto_homing_enabled(False)
        self.update_sensor_count = 0
        self.update_sensor_data()

    def move_to_pose(self, distination: Pose):
        ratio = 1000
        self.client.move_to_pose(
            distination.x / 1000, distination.y / 1000, distination.theta
        )

        while client.is_command_running():
            pass

    def is_moving_finished(self) -> bool:
        return not self.client.is_command_running()

    def update_sensor_data(self):
        self.client.set_manual_control_enabled(True)

        scan = self.client.get_ros_laser_scan()
        theta = np.linspace(scan.angle_min, scan.angle_max, len(scan.ranges))
        dist = np.array(scan.ranges)

        position = self.client.get_robot_pose()
        position = {"x": position.x, "y": position.y, "theta": position.theta}

        ratio = 1000
        self.pose = Pose(
            position["x"] * ratio, position["y"] * ratio, position["theta"]
        )
        self.lidar_data = LidarData(dist, theta, self.pose)

        print(self.pose)
        ax = plt.subplot()
        ax.set_aspect("equal")
        ax.scatter(dist * np.cos(theta), dist * np.sin(theta))

        # ファイルをバイナリモードで開く
        # オブジェクトをシリアライズしてファイルに書き込む
        with open(
            "sensor_log/theta{}.pkl".format(self.update_sensor_count), "wb"
        ) as file:
            pickle.dump(theta, file)

        with open(
            "sensor_log/dist{}.pkl".format(self.update_sensor_count), "wb"
        ) as file:
            pickle.dump(dist, file)

        with open(
            "sensor_log/position{}.pkl".format(self.update_sensor_count), "wb"
        ) as file:
            pickle.dump(position, file)

        self.update_sensor_count += 1
