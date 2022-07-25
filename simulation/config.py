import dataclasses
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
    camera_id: int = 3  # {"workbench_camera": 0, "upper_camera": 1, "gripper_camera": 2, "all": 3}
    show_obs: bool = True

    # actuator parameters
    max_rotation: float = 0.15
    max_translation: float = 0.05
    grasp_tolerance: float = 0.03
    pos_tolerance: float = 0.002
    include_roll: bool = True


    # simulation parameters
    max_steps: int = 1000

    # Markov decision process parameters
    discount_factor: float = 0.99
    time_horizon: int = 200
#
# if __name__ == '__main__':
#     config = Config()
#     print(config)
