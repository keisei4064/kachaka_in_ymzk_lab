from dataclasses import dataclass
from typing import List, Callable
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes
from matplotlib.patches import Rectangle, Circle, FancyArrow
import abc
from enum import Enum
import math
import sys
from IPython.display import display, Image, clear_output
import os
import shutil
from PIL import Image as PILImage


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
    """x,y座標に変換したLiDARの点群データ"""

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
        """LiDARのデータから障害物のあるグリッドを検出し, マップを構成するグリッドが通過可能かどうかを更新する

        Args:
            lidar_data (LidarData): LiDARのデータ
        """
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
    size = Size(160, 235)

    def __init__(self, initial_pose: Pose):
        self.color = BoxColor.UNKNOWN
        self.pose = initial_pose

    def draw(self, ax: matplotlib.axes.Axes):
        rect = make_rectangle_to_center(
            self.pose, Box.size, Box.plot_color[self.color], True, "Box"
        )
        ax.add_patch(rect)


class PushingBoxStatus(Enum):
    PUSHING = 0
    DROPPED_LEFT = 1
    DROPPED_RIGHT = 2


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
        self.is_pushing_box_flag = False  # 現在箱を押しているかどうか

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

        # LiDAR点群の描画
        ax.scatter(
            self.lidar_data.x_data,
            self.lidar_data.y_data,
            color=KachakaBase.lidar_points_color,
            marker=".",
        )

    @abc.abstractmethod
    def move_to_pose(self, distination: Pose) -> None:
        """指定した姿勢まで移動する．（移動が完了するまで処理はブロック）

        Args:
            distination (Pose): 目標姿勢
        """
        pass

    def is_collition_with_box(self, box_pose: Pose) -> bool:
        dist = distance(self.pose, box_pose)
        shortest_dist = KachakaBase.box_size.width / 2 + Box.size.height / 2
        if dist < shortest_dist:
            return True
        else:
            return False

    def move_to_pose_with_box(self, distination: Pose, box: Box) -> None:
        """箱と共に移動する

        Args:
            distination (Pose): 目標姿勢
            box (Box): 箱
        """
        prev_pose = self.pose
        self.move_to_pose(distination)
        now_pose = self.pose
        delta_coordinate = Coordinate(
            now_pose.x - prev_pose.x, now_pose.y - prev_pose.y
        )

        # 箱との当たり判定
        self.is_pushing_box_flag = False
        count = 0
        while self.is_collition_with_box(box.pose) and count < 15:
            self.is_pushing_box_flag = True
            # 箱の移動
            box.pose.x += delta_coordinate.x / 8.0
            box.pose.y += delta_coordinate.y / 8.0
            count += 1

    def is_pushing_box(self) -> bool:
        """箱を押しているかどうかを返す

        Returns:
            bool: 箱を押している場合True, そうでない場合False
        """
        return self.is_pushing_box_flag

    @abc.abstractmethod
    def update_sensor_data(self) -> None:
        """センサデータ (LiDAR, 自己位置，カメラなど) を更新する"""

    @abc.abstractmethod
    def recognize_box_color(self) -> BoxColor:
        """箱の色を認識する

        Returns:
            BoxColor: 認識した色
        """
        pass

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

    @abc.abstractmethod
    def check_pushing_box_by_camera(self) -> PushingBoxStatus:
        """カメラで箱を押しているかどうかを確認する

        Returns:
            PushingBoxStatus: PUSHING: 箱を押せている場合, DROPPED_LEFT: 箱が左に落ちている場合, DROPPED_RIGHT: 箱が右に落ちている場合
        """
        pass


class Trajectory:
    """経路を表すクラス．実質的には目標姿勢のキュー．"""

    color = "orange"

    def __init__(self, trace_poses: list[Pose]):
        self.trace_poses = trace_poses

    def get_next_pose(self) -> Pose:
        """次の目標姿勢を返し，軌道から削除する

        Returns:
            Pose: 次の目標姿勢
        """
        return self.trace_poses.pop(0)

    def is_goal(self) -> bool:
        # 目標姿勢がなくなったらゴール
        return len(self.trace_poses) == 0

    def draw(self, ax: matplotlib.axes.Axes):
        # 矢印の描画
        for pose in self.trace_poses:
            point = Circle((pose.x, pose.y), 5, fill=True, color=Trajectory.color)
            arrow_length = 1
            arrow = FancyArrow(
                pose.x,
                pose.y,
                arrow_length * math.cos(pose.theta),
                arrow_length * math.sin(pose.theta),
                width=0.5,
                head_width=50,
                head_length=50,
                color=Trajectory.color,
            )
            ax.add_patch(point)
            ax.add_patch(arrow)

        # 線の描画
        for i in range(len(self.trace_poses) - 1):
            label = "Trajectory" if i == 0 else None
            ax.plot(
                [self.trace_poses[i].x, self.trace_poses[i + 1].x],
                [self.trace_poses[i].y, self.trace_poses[i + 1].y],
                color=Trajectory.color,
                linewidth=1.0,
                label=label,
            )

    def add_trajectory_to_beginning(self, additional_trajectory: list[Pose]):
        """現在の経路のはじめに経路を追加する

        Args:
            additional_trajectory (list[Pose]): 追加経路
        """
        self.trace_poses = additional_trajectory + self.trace_poses

    def add_trajectory_to_end(self, additional_trajectory: list[Pose]):
        """現在の経路の終わりに経路を追加する

        Args:
            additional_trajectory (list[Pose]): 追加経路
        """
        self.trace_poses = self.trace_poses + additional_trajectory

    def extend_beggining_path(self, distance: float):
        move_angle = self.trace_poses[0].theta
        extended_point = Pose(
            self.trace_poses[0].x - distance * math.cos(move_angle),
            self.trace_poses[0].y - distance * math.sin(move_angle),
            move_angle,
        )
        self.trace_poses.insert(0, extended_point)


class TrajectoryPlannerBase(abc.ABC):
    @abc.abstractmethod
    def plan_trajectory(
        self, start: Pose, goal: Pose, map: GridMap, push_box: bool
    ) -> Trajectory:
        """スタート姿勢からゴール姿勢までを結ぶ経路を生成する

        Args:
            start (Pose): スタート姿勢
            goal (Pose): ゴール姿勢
            map (GridMap): マップ情報
            push_box (bool): 箱を押すかどうか

        Returns:
            Trajectory: 生成した経路
        """
        pass

    @staticmethod
    def IsObstacleOnTrajectory(trajectory: Trajectory, map: GridMap) -> bool:
        """障害物が経路上にあるかどうかを判定する

        Args:
            trajectory (Trajectory): 道筋
            map (GridMap): マップ

        Returns:
            bool: 障害物が道の上にある場合True, ない場合False
        """
        for point in list(trajectory.trace_poses):
            index = map.CoordinateToGridIndex(point)
            if map.grids[index[0]][index[1]].can_pass is False:
                return True

        return False


class StraightTrajectoryPlanner(TrajectoryPlannerBase):
    def __init__(self):
        pass

    def plan_trajectory(
        self, start: Pose, goal: Pose, map: GridMap, push_box: bool
    ) -> Trajectory:
        # 分解能となる距離を計算
        resolution_distance = map.grid_size.height * 0.9  # マップの1マスよりは小さくする

        x_move_theta = 0 if start.x < goal.x else math.pi  # x方向移動時の角度
        y_move_theta = math.pi / 2 if start.y < goal.y else -math.pi / 2  # y方向移動時の角度

        if push_box:  # 箱を押す場合
            # x方向の移動点を計算
            x_res = resolution_distance if start.x < goal.x else -resolution_distance
            box_offset = (Box.size.width / 2) + KachakaBase.box_size.height
            if start.x > goal.x:
                box_offset = -box_offset

            x_points = np.arange(start.x, goal.x - box_offset, x_res)
            x_move_trajectory = [
                Pose(x_point, start.y, x_move_theta) for x_point in x_points
            ]
            x_move_trajectory.append(Pose(goal.x - box_offset, start.y, x_move_theta))

            # y方向へ箱をよけつつターン
            start_turn_pose = x_move_trajectory[-1]
            y_target = (
                start.y - Box.size.height - KachakaBase.box_size.height
                if start.y < goal.y
                else start.y + Box.size.height + KachakaBase.box_size.height
            )
            turn_trajectory = [
                Pose(start_turn_pose.x, start_turn_pose.y, -y_move_theta)
            ]
            turn_trajectory.append(Pose(start_turn_pose.x, y_target, -y_move_theta))
            turn_trajectory.append(Pose(goal.x, y_target, x_move_theta))
            turn_trajectory.append(Pose(goal.x, y_target, y_move_theta))

            # y方向の移動点を計算
            y_res = resolution_distance if start.y < goal.y else -resolution_distance

            box_offset = (Box.size.height / 2) + KachakaBase.box_size.height
            if start.y > goal.y:
                box_offset = -box_offset

            y_points = np.arange(start.y, goal.y - box_offset, y_res)
            y_move_trajectory = [
                Pose(goal.x, y_point, y_move_theta) for y_point in y_points
            ]
            y_move_trajectory.append(Pose(goal.x, goal.y - box_offset, y_move_theta))

            # 統合
            whole_trajectory = x_move_trajectory + turn_trajectory + y_move_trajectory

            if StraightTrajectoryPlanner.IsObstacleOnTrajectory(
                Trajectory(whole_trajectory), map
            ):  # 障害物がある場合
                # x方向の移動とy方向の移動の順番を入れ替える
                x_move_trajectory = [
                    Pose(pose.x, goal.y, x_move_theta) for pose in x_move_trajectory
                ]
                y_move_trajectory = [
                    Pose(start.x, pose.y, y_move_theta) for pose in y_move_trajectory
                ]

                # x方向へ箱をよけつつターン
                start_turn_pose = y_move_trajectory[-1]
                x_target = (
                    start.x - Box.size.width - KachakaBase.box_size.height
                    if start.x < goal.x
                    else start.x + Box.size.width + KachakaBase.box_size.height
                )
                turn_trajectory.clear()
                turn_trajectory = [
                    Pose(start_turn_pose.x, start_turn_pose.y, x_move_theta + math.pi)
                ]
                turn_trajectory.append(
                    Pose(x_target, start_turn_pose.y, x_move_theta + math.pi)
                )
                turn_trajectory.append(Pose(x_target, goal.y, y_move_theta))
                turn_trajectory.append(Pose(x_target, goal.y, x_move_theta))

                whole_trajectory = (
                    y_move_trajectory + turn_trajectory + x_move_trajectory
                )

        else:  # 箱を押さない場合
            print("箱を押さない場合は未実装です")
            sys.exit("異常終了")

        return Trajectory(whole_trajectory)


class CurveTrajectoryPlanner(TrajectoryPlannerBase):
    def __init__(self):
        pass


class ILogger(abc.ABC):
    @abc.abstractmethod
    def log(self, message: str) -> None:
        pass


class TextLogger(ILogger):
    def __init__(self):
        self.history: list[str] = []

    def log(self, message: str) -> None:
        self.history.append(message)
        print(message)

    def show_history(self):
        for message in self.history:
            print(message)


def calc_kachaka_angle_from_box(kachaka_pose: Pose, box: Box) -> float:
    """箱から見てカチャカがどの角度にあるか計算する

    Args:
        kachaka (_type_): カチャカ
        box (_type_): 箱

    Returns:
        float: 角度（π ~ -π）
    """
    dx = kachaka_pose.x - box.pose.x
    dy = kachaka_pose.y - box.pose.y
    return math.atan2(dy, dx)


def clamp_to_pi_range(value: float):
    while value > math.pi:
        value -= 2 * math.pi
    while value < -math.pi:
        value += 2 * math.pi
    return value


class Controller:
    def __init__(
        self,
        kachaka: KachakaBase,
        box: Box,
        map: GridMap,
        trajectory_planner: TrajectoryPlannerBase,
        logger: ILogger,
    ):
        self.kachaka = kachaka
        self.box = box
        self.map = map
        self.trajectory_planner = trajectory_planner
        self.logger = logger
        self.trajectory: Trajectory = Trajectory([])
        # タスクキュー. タスクが完了したらTrueを返す関数を格納する
        self.task_queue: List[Callable[[], bool]] = [
            lambda: self.initialize_sensor(),
            lambda: self.move_from_start_to_initial_box_pose(),
            lambda: self.follow_trajectory(),
            lambda: self.color_recognize(),
            lambda: self.plan_trajectory_of_carrying_box(),
            lambda: self.plan_trajectory_from_now_position_to_carrying_box_start(),
            lambda: self.follow_trajectory(),
            lambda: self.finish_task(),
        ]
        # 箱に触れない距離
        self.no_touching_box_rudius = Box.size.width + KachakaBase.box_size.height

    def initialize_sensor(self) -> bool:
        # self.logger.log("センサ情報を更新します")
        self.kachaka.update_sensor_data()
        lidar_data = self.kachaka.get_lidar_data()
        self.map.DetectObstacleZone(lidar_data)
        return True

    def move_from_start_to_initial_box_pose(self) -> bool:
        self.logger.log("箱の前まで移動します")
        # マップから見て箱の左側に移動
        self.trajectory = Trajectory(
            [
                self.kachaka.get_pose(),
                Pose(
                    self.map.initial_box_pose.x - self.no_touching_box_rudius,
                    self.map.initial_box_pose.y,
                    0,
                ),
            ]
        )
        return True

    def color_recognize(self) -> bool:
        self.logger.log("箱の色を読みとります")
        # 色認識処理------------
        self.box.color = self.kachaka.recognize_box_color()
        # ---------------------
        if self.box.color is BoxColor.BLUE:
            self.logger.log("   青の箱を検出しました")
        elif self.box.color is BoxColor.RED:
            self.logger.log("   赤の箱を検出しました")
        else:
            self.logger.log("   色の検出に失敗しました")

        return True

    def plan_trajectory_of_carrying_box(self) -> bool:
        self.logger.log("箱を運ぶ経路を生成します")
        goal = None
        if self.box.color is BoxColor.BLUE:
            goal = self.map.blue_box_goal
        elif self.box.color is BoxColor.RED:
            goal = self.map.red_box_goal
        else:
            self.logger.log("箱の色が不明なため経路を生成できません")
            sys.exit("異常終了")

        # 経路生成
        self.trajectory = self.trajectory_planner.plan_trajectory(
            self.box.pose, goal, self.map, True
        )

        return True

    def plan_trajectory_from_now_position_to_carrying_box_start(self) -> bool:
        self.trajectory.extend_beggining_path(self.no_touching_box_rudius)

        now_angle_from_box = calc_kachaka_angle_from_box(self.kachaka.pose, self.box)
        target_angle_from_box = calc_kachaka_angle_from_box(
            self.trajectory.trace_poses[0], self.box
        )
        move_angle = target_angle_from_box - now_angle_from_box  # 回転する角度
        clamped_move_angle = clamp_to_pi_range(move_angle)
        # self.logger.log(f"{move_angle=}")

        # 箱を避けながら押し始める位置まで円状に移動する経路を生成
        additional_trajectory: list[Pose] = []
        if clamped_move_angle < 0:  # 箱に対して時計周りに動く
            angles = np.arange(
                now_angle_from_box,
                now_angle_from_box + clamped_move_angle,
                math.radians(-45),
            )
            for angle in angles:
                additional_trajectory.append(
                    Pose(
                        self.box.pose.x + self.no_touching_box_rudius * math.cos(angle),
                        self.box.pose.y + self.no_touching_box_rudius * math.sin(angle),
                        angle - math.pi / 2,
                    )
                )
        elif clamped_move_angle > 0:  # 箱に対して反時計回りに動く
            angles = np.arange(
                now_angle_from_box,
                now_angle_from_box + clamped_move_angle,
                math.radians(+45),
            )
            for angle in angles:
                additional_trajectory.append(
                    Pose(
                        self.box.pose.x + self.no_touching_box_rudius * math.cos(angle),
                        self.box.pose.y + self.no_touching_box_rudius * math.sin(angle),
                        angle + math.pi / 2,
                    )
                )
        else:  # 移動する必要がない場合
            # self.logger.log("no need to rotate")
            pass

        # 角度を-π ~ πの範囲に収める
        for i in range(len(additional_trajectory)):
            additional_trajectory[i].theta = clamp_to_pi_range(
                additional_trajectory[i].theta
            )

        # self.logger.log(f"{additional_trajectory}")

        self.trajectory.add_trajectory_to_beginning(additional_trajectory)
        self.logger.log("経路を辿ります")
        return True

    def follow_trajectory(self) -> bool:
        # ゴール判定
        if self.trajectory.is_goal():
            return True

        # 経路を辿る
        self.kachaka.move_to_pose_with_box(self.trajectory.get_next_pose(), self.box)

        if self.kachaka.is_pushing_box():  # 経路的に箱を押しているはず
            # カメラで本当に箱を押しているか確認
            diff_theta = 0
            pushing_status = self.kachaka.check_pushing_box_by_camera()
            if pushing_status is PushingBoxStatus.DROPPED_LEFT:
                self.logger.log("箱を左に落としました．")
                diff_theta = math.radians(30)
            elif pushing_status is PushingBoxStatus.DROPPED_RIGHT:
                self.logger.log("箱を右に落としました．")
                diff_theta = math.radians(-30)
            else:
                # 箱が問題なく押せている
                return False

            # 箱位置をずらす
            dist = distance(self.kachaka.pose, self.box.pose)
            self.box.pose = Pose(
                self.kachaka.pose.x
                + dist * math.cos(self.kachaka.pose.theta + diff_theta),
                self.kachaka.pose.y
                + dist * math.sin(self.kachaka.pose.theta + diff_theta),
                0,
            )
            self.kachaka.is_pushing_box_flag = False  # 押せてません

            # 経路修正
            self.logger.log("   経路を修正します")

            # 経路生成フェーズからやり直す
            self.task_queue = [
                lambda: self.plan_trajectory_of_carrying_box(),
                lambda: self.plan_trajectory_from_now_position_to_carrying_box_start(),
            ] + self.task_queue

        return False

    def finish_task(self) -> bool:
        self.logger.log("タスクが完了しました")
        return True

    def update(self) -> None:
        self.kachaka.update_sensor_data()

        # LiDARデータから障害物情報を更新
        lidar_data = self.kachaka.get_lidar_data()
        self.map.DetectObstacleZone(lidar_data)

        # タスクの実行
        result = self.task_queue[0]()
        if result:
            self.task_queue.pop(0)

    def are_all_tasks_done(self) -> bool:
        """全てのタスクが完了したかどうかを返す

        Returns:
            bool: 完了時True, 未完了時False
        """
        return len(self.task_queue) == 0


class Plotter:
    def __init__(
        self,
        x_lim: tuple[float, float],
        y_lim: tuple[float, float],
        output_dir: str = "./plotter_output",
    ) -> None:
        self.fig = plt.figure()
        self.ax = plt.subplot()
        self.ax.set_aspect("equal")
        self.x_lim = x_lim
        self.y_lim = y_lim
        self.image_count = 0
        self.output_dir = output_dir
        # 出力ディレクトリが存在する場合、削除して再作成
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
        os.makedirs(self.output_dir)

    def update(
        self,
        map: GridMap,
        box: Box,
        kachaka: KachakaBase,
        trajectory: Trajectory,
    ):
        plt.cla()
        map.draw(self.ax)
        kachaka.draw(self.ax)
        box.draw(self.ax)
        trajectory.draw(self.ax)

        self.ax.set_xlim(*self.x_lim)
        self.ax.set_ylim(*self.y_lim)
        plt.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))

        # 画像を保存
        path = os.path.join(self.output_dir, f"figure_{self.image_count}.png")
        self.fig.savefig(path)
        # 表示をクリア
        clear_output(wait=True)
        # 画像を表示
        display(Image(filename=path))
        self.image_count += 1

    def make_gif(self, duration: int = 200):
        image_list = [
            PILImage.open(os.path.join(self.output_dir, f"figure_{i}.png"))
            for i in range(self.image_count)
        ]
        image_list[0].save(
            self.output_dir + "/animation.gif",
            save_all=True,
            append_images=image_list[1:],
            duration=duration,
            loop=0,
        )

    def make_zip_package(self):
        # もし既にzipファイルが存在する場合は削除
        if os.path.exists("plotter_output.zip"):
            os.remove("plotter_output.zip")

        shutil.make_archive("plotter_output", "zip", self.output_dir)

    def close(self):
        plt.close()
