from dataclasses import dataclass
import numpy as np
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
        lidar_angle = raw_data_theta + diff_angle + offset_pose.theta
        # lidar_angle = raw_data_theta + diff_angle

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
        self.lidar_data = LidarData(np.array([]), np.array([]), self.pose)

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
            self.lidar_data.x_data,
            self.lidar_data.y_data,
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
    def update_sensor_data(self) -> None:
        """センサデータ (LiDAR, 自己位置，カメラなど) を更新する"""

    def get_pose(self) -> Pose:
        """現在の姿勢を返す

        Returns:
            Pose: 現在の姿勢
        """
        return self.pose

    def get_lidar_data(self) -> LidarData:
        """座標に変換したLiDARデータを返す

        Returns:
            LidarData: 座標に変換したLiDARデータ
        """
        return self.lidar_data


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
    size = Size(150, 200)

    def __init__(self, initial_pose: Pose):
        self.color = BoxColor.UNKNOWN
        self.pose = initial_pose

    def draw(self, ax: matplotlib.axes.Axes):
        rect = make_rectangle_to_center(
            self.pose, Box.size, Box.plot_color[self.color], True, "Box"
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
    def plan_path(self, start: Pose, goal: Pose, map: GridMap) -> Path:
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

    def plan_path(self, start: Pose, goal: Pose, map: GridMap) -> Path:
        # 分解能となる距離を計算
        resolution_distance = (
            math.sqrt(map.grid_size.width**2 + map.grid_size.height**2) / 2
        )
        x_move_theta = 0 if start.x < goal.x else math.pi  # x方向移動時の角度
        y_move_theta = math.pi / 2 if start.y < goal.y else -math.pi / 2  # y方向移動時の角度

        # x方向の移動点を計算
        x_res = resolution_distance if start.x < goal.x else -resolution_distance
        box_offset = Box.size.width if start.x < goal.x else -Box.size.width
        x_points = np.arange(start.x, goal.x - box_offset, x_res)
        x_move_path = [Pose(x_point, start.y, x_move_theta) for x_point in x_points]
        x_move_path.append(Pose(goal.x - box_offset, start.y, x_move_theta))

        # y方向へ箱をよけつつターン
        start_turn_pose = x_move_path[-1]
        y_target = (
            start.y - Box.size.height if start.y < goal.y else start.y + Box.size.height
        )
        turn_path = [Pose(start_turn_pose.x, start_turn_pose.y, -y_move_theta)]
        turn_path.append(Pose(start_turn_pose.x, y_target, -y_move_theta))
        turn_path.append(Pose(goal.x, y_target, x_move_theta))
        turn_path.append(Pose(goal.x, y_target, y_move_theta))

        # y方向の移動点を計算
        y_res = resolution_distance if start.y < goal.y else -resolution_distance
        box_offset = Box.size.height if start.y < goal.y else -Box.size.height
        y_points = np.arange(start.y, goal.y - box_offset, y_res)
        y_move_path = [Pose(goal.x, y_point, y_move_theta) for y_point in y_points]
        y_move_path.append(Pose(goal.x, goal.y - box_offset, y_move_theta))

        # 統合
        whole_path = x_move_path + turn_path + y_move_path

        if StraightPathPlanner.IsObstacleOnPath(Path(whole_path), map):  # 障害物がある場合
            # x方向の移動とy方向の移動の順番を入れ替える
            x_move_path = [Pose(pose.x, goal.y, x_move_theta) for pose in x_move_path]
            y_move_path = [Pose(start.x, pose.y, y_move_theta) for pose in y_move_path]

            # x方向へ箱をよけつつターン
            start_turn_pose = y_move_path[-1]
            x_target = (
                start.x - Box.size.width
                if start.x < goal.x
                else start.x + Box.size.width
            )
            turn_path.clear()
            turn_path = [
                Pose(start_turn_pose.x, start_turn_pose.y, x_move_theta + math.pi)
            ]
            turn_path.append(Pose(x_target, start_turn_pose.y, x_move_theta + math.pi))
            turn_path.append(Pose(x_target, goal.y, y_move_theta))
            turn_path.append(Pose(x_target, goal.y, x_move_theta))

            whole_path = y_move_path + turn_path + x_move_path

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
        box: Box,
        map: GridMap,
        path_planner: PathPlannerBase,
        logger: ILogger,
    ):
        self.kachaka = kachaka
        self.box = box
        self.map = map
        self.path_planner = path_planner
        self.logger = logger
        self.path: Path = Path([])
        self.action = lambda: self.initialize_sensor()

    def initialize_sensor(self) -> None:
        self.logger.log("センサ情報を更新します")
        self.kachaka.update_sensor_data()
        lidar_data = self.kachaka.get_lidar_data()
        self.map.DetectObstacleZone(lidar_data)

        self.action = lambda: self.move_from_start_to_initial_box_pose()

    def move_from_start_to_initial_box_pose(self) -> None:
        self.logger.log("箱の前まで移動します")
        self.kachaka.move_to_pose(
            Pose(
                self.map.initial_box_pose.x,
                self.map.initial_box_pose.y - Box.size.height,
                0,
            )
        )

        self.action = lambda: self.color_recognize()

    def color_recognize(self) -> None:
        self.logger.log("箱の色を読みとります")
        # 色認識処理------------
        self.box.color = BoxColor.BLUE
        # ---------------------
        if self.box.color is BoxColor.BLUE:
            self.logger.log("青の箱を検出しました")
        elif self.box.color is BoxColor.RED:
            self.logger.log("赤の箱を検出しました")
        else:
            self.logger.log("色の検出に失敗しました")

        self.action = lambda: self.plan_path()

    def plan_path(self):
        self.logger.log("箱を運ぶ経路を生成します")
        goal = None
        if self.box.color is BoxColor.BLUE:
            goal = self.map.blue_box_goal
        elif self.box.color is BoxColor.RED:
            goal = self.map.red_box_goal

        if goal is not None:
            self.path = self.path_planner.plan_path(
                self.map.initial_box_pose, goal, self.map
            )

    def move_from_initial_box_pose_to_goal(self) -> None:
        self.logger.log("Start moving from initial box pose to goal")

    def update(self) -> None:
        self.action()


class Plotter:
    def __init__(
        self,
        x_lim: tuple[float, float],
        y_lim: tuple[float, float],
    ) -> None:
        self.fig = plt.figure()
        self.ax = plt.subplot()
        self.ax.set_aspect("equal")
        self.x_lim = x_lim
        self.y_lim = y_lim

    def update(
        self,
        map: GridMap,
        box: Box,
        kachaka: KachakaBase,
        path: Path,
    ):
        plt.cla()
        map.draw(self.ax)
        kachaka.draw(self.ax)
        box.draw(self.ax)
        path.draw(self.ax)
        self.ax.set_xlim(*self.x_lim)
        self.ax.set_ylim(*self.y_lim)
        plt.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
        plt.pause(0.01)