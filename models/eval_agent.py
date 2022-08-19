import os
from pathlib import Path
from models.feature_extractor import AugmentedNatureCNN
from scripts.plot_3D import plot_3D
from simulation.environment.robot_env import RobotEnv
from config.eval_config import EvalConfig
from stable_baselines3 import SAC
from PIL import Image


def make_gif(frames_in):
    """
    Save a gif of the trajectory based on the frames "frames_in"
    """
    # check if the directory exists, if not create it
    gif_dir = Path(__file__).resolve().parent.parent.as_posix() + "/visuals/gifs/"
    if not os.path.exists(gif_dir):
        os.makedirs(gif_dir)

    # load the frames into a PIL image object and save it as a gif
    frames = [Image.fromarray(image) for image in frames_in]
    frame_one = frames[0]
    frame_one.save(os.path.join(gif_dir, f"{config.train_env}_on_{config.sim_env.split('/')[-1].split('.')[0]}_dir{config.direction}.gif"),
                   format="GIF", append_images=frames,
                   save_all=True, duration=100, loop=0)


def evaluate():
    """
    Evaluate the trained model on the evaluation environment
    """
    env = RobotEnv(config)

    model_dir = Path(__file__).resolve().parent.parent.as_posix() + "/models/trained_models/"
    model_path = f"dir{config.direction}_reward/{config.train_env}_{config.reward_type}_reward_best_model"

    best_model = os.path.join(model_dir, model_path, 'best_model.zip')

    newer_python_version = True
    custom_objects = {}
    if newer_python_version:
        custom_objects = {
            "policy_kwargs": dict(features_extractor_class=AugmentedNatureCNN,
                                  share_features_extractor=True,
                                  net_arch=[256, 256]),
        }

    model = SAC.load(best_model, env=env, custom_objects=custom_objects)

    total_step = []
    line_step = []
    robot_step = []
    obj_step = []
    frames = []
    obs = env.reset()
    for i in range(config.max_steps):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        total_step.append(info["total_distance"])
        line_step.append(info["line_distance"])
        robot_step.append(info["gripper_position"].copy())
        obj_step.append(info["object_position"].copy())
        if config.render:
            frames.append(env.render(mode='rgb_array'))
            env.render()
        if dones:
            break

    print(f"Total distance travelled by the object:{sum(total_step):.2f}")
    print(f"Distance travelled on a straight line:{sum(line_step):.2f}, {(sum(line_step)/sum(total_step))*100:.2f}%")

    if config.plot_trajectory:
        plot_3D(robot_step, obj_step, config.sim_env)

    if config.save_gif:
        make_gif(frames)


if __name__ == "__main__":
    config_class = EvalConfig()
    config = config_class.parse()
    config.sim_env = "/xmls/bread_crumb_env.xml"
    config.train_env = "bread_crumb"
    config.reward_type = "im"
    config.direction = 45
    config.render = True
    config.plot_trajectory = True
    config.save_gif = True

    evaluate()
