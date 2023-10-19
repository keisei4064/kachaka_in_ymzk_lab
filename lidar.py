from dataclasses import dataclass
import numpy as np
import pickle
import matplotlib.pyplot as plt
import abc
import queue


@dataclass
class Pose:
    """姿勢を表すデータクラス

    Attributes:
        x (float): x座標
        y (float): y座標
        theta (float): 角度

    """

    x: float
    y: float
    theta: float


class IKachaka(abc.ABC):
    """カチャカのインターフェースクラス"""

    @abc.abstractmethod
    def move_to_pose(self, distination: Pose) -> None:
        """指定した姿勢まで移動する

        Args:
            distination (Pose): 目標姿勢
        """
        pass

    @abc.abstractmethod
    def is_moving_finished(self) -> bool:
        """移動が完了したかどうかを返す

        Returns:
            bool: 完了時True, 未完了時False
        """
        pass

    @abc.abstractmethod
    def get_pose(self) -> Pose:
        """現在の姿勢を返す

        Returns:
            Pose: 現在の姿勢
        """
        pass

    @abc.abstractmethod
    def get_lidar_data(self) -> tuple[np.ndarray, np.ndarray]:
        """LiDARのデータを距離と角度をセットにして返す

        Returns:
            tuple[np.ndarray, np.ndarray]: [距離リスト，角度リスト]
        """
        pass


class VirtualKachaka(IKachaka):
    def __init__(self, data_num="1"):
        self.path = "data/robot_data" + data_num + "/"

        with open(self.path + "position.pkl", "rb") as file:
            position = pickle.load(file)
        self.pose = Pose(position[0], position[1], position[2])

    def move_to_pose(self, distination: Pose):
        pass

    def is_moving_finished(self):
        pass

    def get_pose(self):
        return self.pose

    def get_lidar_data(self, offset_position) -> tuple[np.ndarray, np.ndarray]:
        # データ取得処理...
        data_num = "3"
        path = "data/sensor_data" + data_num + "/"

        with open(path + "theta.pkl", "rb") as file:
            theta = pickle.load(file)

        with open(path + "dist.pkl", "rb") as file:
            dist = pickle.load(file)

        return (dist * np.cos(theta), dist * np.sin(theta))


@dataclass
class Grid:
    """マップの1マスを表すデータクラス

    Attributes:
        can_pass (bool): 通行可能かどうか
    """

    can_pass: bool


class GridMap:
    def __init__(self):
        pass


class Path:
    def __init__(self, trace_poses: list[Pose]):
        self.trace_poses = queue.LifoQueue()
        for pose in trace_poses:
            self.trace_poses.put(pose)

    def get_next_pose(self) -> Pose:
        return self.trace_poses.get()

    def is_goal(self) -> bool:
        return self.trace_poses.empty()


class IPathPlanner(abc.ABC):
    @abc.abstractmethod
    def plan_path(self, start: Pose, goal: Pose) -> Path:
        pass


class StraightPathPlanner(IPathPlanner):
    def __init__(self):
        pass


class CurvePathPlanner(IPathPlanner):
    def __init__(self):
        pass


class Controller:
    def __init__(self, kachaka: IKachaka, map, logger):
        self.kachaka = kachaka
        self.map = map
        self.logger = logger


if __name__ == "__main__":
    lidar = VirtualLidar(1, 2)
    data = lidar.get_lidar_points(1, 2)

    ax = plt.subplot()
    ax.set_aspect("equal")
    ax.scatter(*data)
    plt.show()
