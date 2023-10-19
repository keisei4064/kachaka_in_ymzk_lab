from dataclasses import dataclass
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.axes
from matplotlib.patches import Rectangle, Circle, FancyArrow
import abc
import queue
from enum import Enum
import math


@dataclass
class Pose:
    """姿勢を表すデータクラス

    Attributes:
        x (float): x座標 [mm]
        y (float): y座標 [mm]
        theta (float): 角度 [rad]

    """

    x: float
    y: float
    theta: float


@dataclass
class Size:
    """大きさを表すデータクラス

    Attributes:
        width (float): 幅(x方向) [mm]
        height (float): 高さ(y方向) [mm]

    """

    width: float
    height: float


def distance(pose1: Pose, pose2: Pose) -> float:
    """2点間の距離を求める

    Args:
        pose1 (Pose): 始点
        pose2 (Pose): 終点

    Returns:
        float: 2点間の距離 [mm]
    """
    return math.sqrt((pose1.x - pose2.x) ** 2 + (pose1.y - pose2.y) ** 2)


def make_rectangle_to_center(
    pose: Pose, size: Size, color, fill: bool, label: str | None = None
) -> Rectangle:
    """中心座標とサイズから四角形を作成する

    Args:
        pose (Pose): 描写する座標
        size (Size): 大きさ
        color (_type_): 色
        fill (bool): 塗りつぶすか否か

    Returns:
        Rectangle: 作成した四角形
    """
    distance = math.sqrt(size.width**2 + size.height**2) / 2
    theta = math.atan2(size.height, size.width)
    center_pose = Pose(
        pose.x - distance * math.cos(pose.theta + theta),
        pose.y - distance * math.sin(pose.theta + theta),
        pose.theta,
    )
    return Rectangle(
        xy=(
            center_pose.x,
            center_pose.y,
        ),
        width=size.width,
        height=size.height,
        angle=math.degrees(pose.theta),
        color=color,
        fill=fill,
        label=label,
    )


@dataclass
class Grid:
    """マップの1マスを表すデータクラス

    Attributes:
        can_pass (bool): 通行可能かどうか
    """

    can_pass: bool


class GridMap:
    frame_color = "black"
    start_color = "lightgreen"
    red_box_goal_color = "lightpink"
    blue_box_goal_color = "lightblue"
    start_zone_size = Size(400, 400)
    goal_zone_size = Size(250, 250)

    def __init__(
        self,
        size: Size,
        grid_size: Size,
        origin_offset: Pose,
        start: Pose,
        red_box_goal: Pose,
        blue_box_goal: Pose,
    ):
        self.size = size
        self.grid_size = grid_size
        self.origin_offset = origin_offset

        # 表示用の四角形を作成
        self.map_frame = Rectangle(
            (-origin_offset.x, -origin_offset.y),
            self.size.width,
            self.size.height,
            fill=False,
            color=GridMap.frame_color,
            linewidth=1,
        )
        self.start_zone = make_rectangle_to_center(
            start, GridMap.start_zone_size, GridMap.start_color, False, "Start"
        )
        self.red_box_goal_zone = make_rectangle_to_center(
            red_box_goal,
            GridMap.goal_zone_size,
            GridMap.red_box_goal_color,
            False,
            "Red Goal",
        )
        self.blue_box_goal_zone = make_rectangle_to_center(
            blue_box_goal,
            GridMap.goal_zone_size,
            GridMap.blue_box_goal_color,
            False,
            "Blue Goal",
        )

    def draw(self, ax: matplotlib.axes.Axes):
        ax.add_patch(self.start_zone)
        ax.add_patch(self.red_box_goal_zone)
        ax.add_patch(self.blue_box_goal_zone)
        ax.add_patch(self.map_frame)

    def get_axes_lim(self) -> tuple[tuple[float, float], tuple[float, float]]:
        return (
            (-self.origin_offset.x, self.size.width - self.origin_offset.x),
            (-self.origin_offset.y, self.size.height - self.origin_offset.y),
        )


@dataclass
class LidarData:
    def __init__(
        self, raw_data_dist: np.ndarray, raw_data_theta: np.ndarray, offset_pose: Pose
    ):
        ratio = 1000
        diff_angle = np.pi / 2
        lidar_angle = raw_data_theta + diff_angle
        self.x_data = raw_data_dist * ratio * np.cos(lidar_angle) + offset_pose.x
        self.y_data = raw_data_dist * ratio * np.sin(lidar_angle) + offset_pose.y


class KachakaBase(abc.ABC):
    """カチャカの基底クラス"""

    box_size = Size(387, 240)
    box_color = "silver"
    wheel_size = Size(60, 20)
    wheel_color = "black"
    wheel_interval = 150
    arrow_color = "black"
    lidar_points_color = "darkblue"

    def __init__(self):
        self.pose = Pose(0, 0, 0)
        self.lidar_data_cache = LidarData(np.array([]), np.array([]), self.pose)

    def draw(self, ax: matplotlib.axes.Axes):
        # 車体の描画
        box_rect = make_rectangle_to_center(
            self.pose, KachakaBase.box_size, KachakaBase.box_color, True, "Kachaka"
        )
        left_wheel_rect = make_rectangle_to_center(
            Pose(
                self.pose.x
                + (KachakaBase.box_size.width / 4) * math.sin(self.pose.theta),
                self.pose.y
                - (KachakaBase.box_size.width / 4) * math.cos(self.pose.theta),
                self.pose.theta,
            ),
            KachakaBase.wheel_size,
            KachakaBase.wheel_color,
            True,
        )
        right_wheel_rect = make_rectangle_to_center(
            Pose(
                self.pose.x
                - (KachakaBase.box_size.width / 4) * math.sin(self.pose.theta),
                self.pose.y
                + (KachakaBase.box_size.width / 4) * math.cos(self.pose.theta),
                self.pose.theta,
            ),
            KachakaBase.wheel_size,
            KachakaBase.wheel_color,
            True,
        )

        # 矢印の描画
        point = Circle(
            (self.pose.x, self.pose.y), 5, fill=True, color=KachakaBase.arrow_color
        )
        arrow_length = 100
        arrow = FancyArrow(
            self.pose.x,
            self.pose.y,
            arrow_length * math.cos(self.pose.theta),
            arrow_length * math.sin(self.pose.theta),
            width=1,
            head_width=50,
            color=KachakaBase.arrow_color,
        )

        # 座標文字列の描画
        text_pose = Pose(self.pose.x + 100, self.pose.y + 100, 0)
        text_content = "(x:{:.0f}, y:{:.0f}, θ:{:.2f})".format(
            self.pose.x, self.pose.y, self.pose.theta
        )

        # 軸に追加
        ax.text(text_pose.x, text_pose.y, text_content, fontsize=8)
        ax.add_patch(box_rect)
        ax.add_patch(left_wheel_rect)
        ax.add_patch(right_wheel_rect)
        ax.add_patch(point)
        ax.add_patch(arrow)

        # 点群の描画
        ax.scatter(
            self.lidar_data_cache.x_data,
            self.lidar_data_cache.y_data,
            color=KachakaBase.lidar_points_color,
            linewidths=1,
        )

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
    def get_raw_lidar_data(self) -> tuple[np.ndarray, np.ndarray]:
        """未処理のLiDARデータ (距離と角度)をセットにして返す

        Returns:
            tuple[np.ndarray, np.ndarray]: [距離リスト，角度リスト]
        """
        pass

    def get_processed_lidar_data(self) -> LidarData:
        """座標に変換したLiDARデータを返す

        Returns:
            LidarData: 座標に変換したLiDARデータ
        """
        self.lidar_data_cache = LidarData(*self.get_raw_lidar_data(), self.pose)
        return self.lidar_data_cache


class VirtualKachaka(KachakaBase):
    def __init__(self, data_num="3"):
        super().__init__()
        self.path = "data/sensor_data" + data_num + "/"
        self.pose = self.get_pose()
        self.get_processed_lidar_data()

    def move_to_pose(self, distination: Pose):
        pass

    def is_moving_finished(self):
        pass

    def get_pose(self):
        with open(self.path + "position.pkl", "rb") as file:
            position = pickle.load(file)

        ratio = 1000

        return Pose(position["x"] * ratio, position["y"] * ratio, position["theta"])

    def get_raw_lidar_data(self) -> tuple[np.ndarray, np.ndarray]:
        # データ取得処理...
        data_num = "3"
        path = "data/sensor_data" + data_num + "/"

        with open(path + "theta.pkl", "rb") as file:
            theta = pickle.load(file)

        with open(path + "dist.pkl", "rb") as file:
            dist = pickle.load(file)

        return (dist, theta)


class BoxColor(Enum):
    UNKNOWN = 0
    RED = 1
    BLUE = 2


class Box:
    """カチャカで運ぶ箱"""

    plot_color = {
        BoxColor.UNKNOWN: "lightgray",
        BoxColor.RED: "red",
        BoxColor.BLUE: "blue",
    }

    def __init__(self, initial_pose: Pose, size: Size):
        self.color = BoxColor.UNKNOWN
        self.pose = initial_pose
        self.size = size

    def draw(self, ax: matplotlib.axes.Axes):
        rect = make_rectangle_to_center(
            self.pose, self.size, Box.plot_color[self.color], True, "Box"
        )
        ax.add_patch(rect)


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


class ILogger(abc.ABC):
    @abc.abstractmethod
    def log(self, message: str) -> None:
        pass


class TextLogger(ILogger):
    def __init__(self):
        pass

    def log(self, message: str) -> None:
        print(message)


class ChatLogger(ILogger):
    def __init__(self):
        pass

    def log(self, message: str) -> None:
        print(message)
        # 喋る処理


class Controller:
    def __init__(self, kachaka: KachakaBase, map: GridMap, logger: ILogger):
        self.kachaka = kachaka
        self.map = map
        self.logger = logger


class Plotter:
    def __init__(
        self,
        x_lim: tuple[float, float],
        y_lim: tuple[float, float],
    ) -> None:
        self.ax = plt.subplot()
        self.ax.set_aspect("equal")
        self.ax.set_xlim(*x_lim)
        self.ax.set_ylim(*y_lim)

    def update(self, map: GridMap, box: Box, kachaka: KachakaBase) -> None:
        map.draw(self.ax)
        box.draw(self.ax)
        kachaka.draw(self.ax)
        plt.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
        plt.show()
        self.ax.cla()


if __name__ == "__main__":
    # 初期化 -------------------------------------------------------------
    initial_box_pose = Pose(1000, 1000, 0)
    red_box_goal = Pose(1500, -500, 0)
    blue_box_goal = Pose(250, 1500, 0)
    box = Box(initial_box_pose, Size(150, 200))
    kachaka = VirtualKachaka("4")
    map = GridMap(
        Size(2340, 2890),
        grid_size=Size(50, 50),
        origin_offset=Pose(400, 950, 0),
        start=Pose(-200, 0, 0),
        red_box_goal=red_box_goal,
        blue_box_goal=blue_box_goal,
    )
    logger = TextLogger()
    controller = Controller(kachaka, map, logger)

    (x_lim, y_lim) = map.get_axes_lim()
    # マージン確保
    margin = 1500
    x_lim = (x_lim[0] - margin, x_lim[1] + margin)
    y_lim = (y_lim[0] - margin, y_lim[1] + margin)
    plotter = Plotter(x_lim, y_lim)

    # メインループ ---------------------------------------------------------
    while True:
        plotter.update(map, box, kachaka)
