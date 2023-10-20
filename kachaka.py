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
class Coordinate:
    """座標を表すデータクラス

    Attributes:
        x (float): x座標 [mm]
        y (float): y座標 [mm]

    """

    x: float
    y: float


@dataclass
class Pose(Coordinate):
    """姿勢を表すデータクラス

    Attributes:
        x (float): x座標 [mm]
        y (float): y座標 [mm]
        theta (float): 角度 [rad]

    """

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


def distance(point1: Coordinate, point2: Coordinate) -> float:
    """2点間の距離を求める

    Args:
        point1 (Coordinate): 始点
        point2 (Coordinate): 終点

    Returns:
        float: 2点間の距離 [mm]
    """
    return math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)


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
        linewidth=1.5,
    )


class LidarData:
    def __init__(
        self, raw_data_dist: np.ndarray, raw_data_theta: np.ndarray, offset_pose: Pose
    ):
        ratio = 1000
        diff_angle = np.pi / 2
        # lidar_angle = raw_data_theta + diff_angle + offset_pose.theta
        lidar_angle = raw_data_theta + diff_angle

        # 0でない要素のインデックスを取得
        non_zero_indices = np.nonzero(raw_data_dist)

        self.x_data = (
            raw_data_dist[non_zero_indices]
            * ratio
            * np.cos(lidar_angle[non_zero_indices])
            + offset_pose.x
        )
        self.y_data = (
            raw_data_dist[non_zero_indices]
            * ratio
            * np.sin(lidar_angle[non_zero_indices])
            + offset_pose.y
        )


@dataclass
class Grid:
    """マップの1マスを表すクラス

    Attributes:
        can_pass (bool): 通行可能かどうか
    """

    color = "lightgray"

    def __init__(self, bottom_left: Coordinate, size: Size, can_pass: bool = True):
        self.can_pass = can_pass
        self.rect = Rectangle(
            xy=(bottom_left.x, bottom_left.y),
            width=size.width,
            height=size.height,
            color=Grid.color,
            fill=not can_pass,
            linewidth=0.5,
        )

    def draw(self, ax: matplotlib.axes.Axes):
        self.rect.set_fill(not self.can_pass)
        ax.add_patch(self.rect)


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
        initial_box_pose: Pose,
        red_box_goal: Pose,
        blue_box_goal: Pose,
    ):
        self.size = size
        self.grid_size = grid_size
        self.origin_offset = origin_offset
        self.start = start
        self.initial_box_pose = initial_box_pose
        self.red_box_goal = red_box_goal
        self.blue_box_goal = blue_box_goal

        # グリッドの作成
        width_num = math.ceil(size.width / grid_size.width) + 2
        height_num = math.ceil(size.height / grid_size.height) + 2
        self.grids: list[list[Grid]] = []
        bottom_left = Coordinate(
            -origin_offset.x - grid_size.width,
            -origin_offset.y - grid_size.height,
        )
        for i in range(height_num):
            row_list = []
            for i in range(width_num):
                row_list.append(Grid(bottom_left, grid_size))
                bottom_left.x += grid_size.width
            self.grids.append(row_list)
            bottom_left.x = -origin_offset.x - grid_size.width
            bottom_left.y += grid_size.height

        for grid in self.grids[0][:]:
            grid.can_pass = False
        for grid in self.grids[-1][:]:
            grid.can_pass = False
        for row in self.grids:
            row[0].can_pass = False
            row[-1].can_pass = False

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
        for row in self.grids:
            for grid in row:
                grid.draw(ax)
        ax.add_patch(self.start_zone)
        ax.add_patch(self.red_box_goal_zone)
        ax.add_patch(self.blue_box_goal_zone)
        ax.add_patch(self.map_frame)

    def get_axes_lim(self) -> tuple[tuple[float, float], tuple[float, float]]:
        return (
            (-self.origin_offset.x, self.size.width - self.origin_offset.x),
            (-self.origin_offset.y, self.size.height - self.origin_offset.y),
        )

    def CoordinateToGridIndex(self, coordinate: Coordinate) -> tuple[int, int]:
        """座標からグリッドのインデックスを返す

        Args:
            coordinate (Coordinate): 座標

        Returns:
            tuple[int, int]: グリッドのインデックス
        """
        x = math.ceil((coordinate.x + self.origin_offset.x) / self.grid_size.width)
        y = math.ceil((coordinate.y + self.origin_offset.y) / self.grid_size.height)
        return (y, x)

    def DetectObstacleZone(self, lidar_data: LidarData) -> None:
        for in_area_row in self.grids[1:-1]:
            for in_area_grid in in_area_row[1:-1]:
                in_area_grid.can_pass = True

        for x, y in zip(lidar_data.x_data, lidar_data.y_data):
            grid_index = self.CoordinateToGridIndex(Coordinate(x, y))
            # マップの範囲内なら
            if (0 < grid_index[1] < len(self.grids[0])) and (
                0 < grid_index[0] < len(self.grids)
            ):
                self.grids[grid_index[0]][grid_index[1]].can_pass = False


class KachakaBase(abc.ABC):
    """カチャカの基底クラス"""

    box_size = Size(387, 240)
    box_color = "gray"
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
            self.pose, KachakaBase.box_size, KachakaBase.box_color, False, "Kachaka"
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
        text_coordinate = Coordinate(self.pose.x + 100, self.pose.y + 100)
        text_content = "(x:{:.0f}, y:{:.0f}, θ:{:.2f})".format(
            self.pose.x, self.pose.y, self.pose.theta
        )

        # 軸に追加
        ax.text(text_coordinate.x, text_coordinate.y, text_content, fontsize=8)
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
            marker=".",
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
        BoxColor.UNKNOWN: "gray",
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
    color = "orange"

    def __init__(self, trace_poses: list[Pose]):
        self.trace_poses = queue.LifoQueue()
        for pose in trace_poses:
            self.trace_poses.put(pose)

    def get_next_pose(self) -> Pose:
        return self.trace_poses.get()

    def is_goal(self) -> bool:
        return self.trace_poses.empty()

    def draw(self, ax: matplotlib.axes.Axes):
        trace_poses = list(self.trace_poses.queue)
        # 矢印の描画
        for pose in trace_poses:
            point = Circle((pose.x, pose.y), 5, fill=True, color=Path.color)
            arrow_length = 1
            arrow = FancyArrow(
                pose.x,
                pose.y,
                arrow_length * math.cos(pose.theta),
                arrow_length * math.sin(pose.theta),
                width=0.5,
                head_width=50,
                head_length=50,
                color=Path.color,
            )
            ax.add_patch(point)
            ax.add_patch(arrow)

        # 線の描画
        for i in range(len(trace_poses) - 1):
            label = "Path" if i == 0 else None
            ax.plot(
                [trace_poses[i].x, trace_poses[i + 1].x],
                [trace_poses[i].y, trace_poses[i + 1].y],
                color=Path.color,
                linewidth=1.0,
                label=label,
            )


class PathPlannerBase(abc.ABC):
    @abc.abstractmethod
    def plan_path(self, box_start: Pose, box_goal: Pose, map: GridMap) -> Path:
        pass

    @staticmethod
    def IsObstacleOnPath(path: Path, map: GridMap) -> bool:
        """障害物が道の上にあるかどうかを判定する

        Args:
            path (Path): 道筋
            map (GridMap): マップ

        Returns:
            bool: 障害物が道の上にある場合True, ない場合False
        """
        for point in list(path.trace_poses.queue):
            index = map.CoordinateToGridIndex(point)
            if map.grids[index[0]][index[1]].can_pass is False:
                return True

        return False


class StraightPathPlanner(PathPlannerBase):
    def __init__(self):
        pass

    def plan_path(self, box_start: Pose, box_goal: Pose, map: GridMap) -> Path:
        # 分解能となる距離を計算
        resolution_distance = (
            math.sqrt(map.grid_size.width**2 + map.grid_size.height**2) / 2
        )

        # x方向の移動点を計算
        x_res = (
            resolution_distance if box_start.x < box_goal.x else -resolution_distance
        )
        x_points = np.arange(box_start.x, box_goal.x, x_res)
        x_move_theta = 0 if box_start.x < box_goal.x else math.pi
        x_move_path = [Pose(x_point, box_start.y, x_move_theta) for x_point in x_points]
        x_move_path.append(Pose(box_goal.x, box_start.y, x_move_theta))

        # y方向の移動点を計算
        y_res = (
            resolution_distance if box_start.y < box_goal.y else -resolution_distance
        )
        y_points = np.arange(box_start.y, box_goal.y, y_res)
        y_move_theta = math.pi / 2 if box_start.y < box_goal.y else -math.pi / 2
        y_move_path = [Pose(box_goal.x, y_point, y_move_theta) for y_point in y_points]
        y_move_path.append(Pose(box_goal.x, box_goal.y, y_move_theta))

        # 統合
        whole_path = x_move_path + y_move_path

        if StraightPathPlanner.IsObstacleOnPath(Path(whole_path), map):  # 障害物がある場合
            # x方向の移動とy方向の移動の順番を入れ替える
            x_move_path = [
                Pose(pose.x, box_goal.y, x_move_theta) for pose in x_move_path
            ]
            y_move_path = [
                Pose(box_start.x, pose.y, y_move_theta) for pose in y_move_path
            ]
            whole_path = y_move_path + x_move_path

        return Path(whole_path)


class CurvePathPlanner(PathPlannerBase):
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
    def __init__(
        self,
        kachaka: KachakaBase,
        map: GridMap,
        path_planner: PathPlannerBase,
        logger: ILogger,
    ):
        self.kachaka = kachaka
        self.map = map
        self.path_planner = path_planner
        self.logger = logger
        self.path: Path = Path([])

    def move_from_initial_box_pose_to_goal(self) -> None:
        self.path = self.path_planner.plan_path(
            self.map.initial_box_pose, self.map.blue_box_goal, self.map
        )

    def draw(self, ax: matplotlib.axes.Axes):
        self.path.draw(ax)


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

    def update(
        self, map: GridMap, box: Box, kachaka: KachakaBase, controller: Controller
    ) -> None:
        map.draw(self.ax)
        kachaka.draw(self.ax)
        box.draw(self.ax)
        controller.draw(self.ax)
        plt.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
        plt.show()
        self.ax.cla()


if __name__ == "__main__":
    # 初期化 -------------------------------------------------------------
    initial_box_pose = Pose(1000, 1000, 0)
    red_box_goal = Pose(1500, -500, 0)
    blue_box_goal = Pose(250, 1500, 0)
    box = Box(initial_box_pose, Size(150, 200))
    kachaka = VirtualKachaka("7")
    map = GridMap(
        Size(2340, 2890),
        grid_size=Size(150, 150),
        origin_offset=Pose(400, 950, 0),
        start=Pose(-200, 0, 0),
        initial_box_pose=initial_box_pose,
        red_box_goal=red_box_goal,
        blue_box_goal=blue_box_goal,
    )
    logger = TextLogger()
    path_planner = StraightPathPlanner()
    controller = Controller(kachaka, map, path_planner, logger)

    (x_lim, y_lim) = map.get_axes_lim()
    # 表示マージン確保
    margin = 1000
    x_lim = (x_lim[0] - margin, x_lim[1] + margin)
    y_lim = (y_lim[0] - margin, y_lim[1] + margin)
    plotter = Plotter(x_lim, y_lim)

    # メインループ ---------------------------------------------------------
    while True:
        lidar_data = kachaka.get_processed_lidar_data()
        map.DetectObstacleZone(lidar_data)
        controller.move_from_initial_box_pose_to_goal()
        plotter.update(map, box, kachaka, controller)
