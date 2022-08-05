import dataclasses
from pathlib import Path


# @dataclasses.dataclass(frozen=True)
class Config:
    """
    Configuration for the simulation.
    """
    xml_path: str = Path(__file__).resolve().parent.parent.as_posix() + r"/xmls/acorn_env.xml"

    # camera parameters
    width_capture: int = 64
    height_capture: int = 64
    rendering_zoom_width: int = 5 * 2
    rendering_zoom_height: int = 3.75 * 2
    full_observation: bool = True
    camera_id: int = 3  # {"workbench_camera": 0, "upper_camera": 1, "gripper_camera": 2, "all": 3}
    show_obs: bool = False

    # actuator parameters
    max_rotation: float = 0.15
    max_translation: float = 0.05
    grasp_tolerance: float = 0.03
    pos_tolerance: float = 0.002
    include_roll: bool = True

    # simulation parameters
    max_steps: int = 400
    im_reward: bool = False
    restart_obj_pos: None
    her_buffer: bool = False

    # Markov decision process parameters
    discount_factor: float = 0.99
    time_horizon: int = 200
    target_direction: None

# if __name__ == '__main__':
#     config = Config()
#     print(config)
