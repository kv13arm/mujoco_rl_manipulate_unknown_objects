import dataclasses
import numpy as np
from pathlib import Path


@dataclasses.dataclass(frozen=True)
class Config:
    """
    Configuration for the simulation.
    """
    seed: int = 42  # OK
    xml_path: str = Path(__file__).resolve().parent.parent.as_posix() + r"/xmls/gripper_two_fingers.xml"  # OK

    # camera parameters
    width_capture: int = 64  # OK
    height_capture: int = 64  # OK
    rendering_zoom_width: int = 5 * 2  # OK
    rendering_zoom_height: int = 3.75 * 2  # OK
    full_observation: bool = True
    camera_id: int = 3 # {"workbench_camera": 0, "upper_camera": 1, "gripper_camera": 2, "all": 3}
    show_obs: bool = True

    # actuator parameters
    max_rotation: float = 0.15
    max_translation: float = 0.05
    # action_space_discrete: bool = False
    # step_size: float = 0.01
    grasp_tolerance: float = 0.2
    gripper_tolerance: float = 0.03
    include_roll: bool = True

    # Markov decision process parameters
    discount_factor: float = 0.99
    time_horizon: int = 200





#     inner_step: int = 300
#
#     viewer_x_coordinate: float = 0
#     viewer_y_coordinate: float = 0
#     viewer_z_coordinate: float = 0.15
#     viewer_elevation: float = -45
#     viewer_azimuth: float = 180
#     viewer_distance_rate: float = 0.
#
#
#
#     def viewer_param(self):
#         param = {}
#         param["viewer_x_coordinate"] = self.viewer_x_coordinate
#         param["viewer_y_coordinate"] = self.viewer_y_coordinate
#         param["viewer_z_coordinate"] = self.viewer_z_coordinate
#         param["viewer_elevation"] = self.viewer_elevation
#         param["viewer_azimuth"] = self.viewer_azimuth
#         param["viewer_distance_rate"] = self.viewer_distance_rate
#         return param
#
#
# if __name__ == '__main__':
#     config = Config()
#     print(config)
#     print(config.viewer_param())
