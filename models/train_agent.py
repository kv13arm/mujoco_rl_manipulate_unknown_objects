from config.train_config import TrainConfig
from simulation.environment.robot_env import RobotEnv
from models.feature_extractor import AugmentedNatureCNN
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
import numpy as np
import time


def train(config):
    env = RobotEnv(config)


    policy_kwargs = dict(features_extractor_class=AugmentedNatureCNN,
                         share_features_extractor=True)
    env = make_vec_env(lambda: env, n_envs=1)

    model = SAC("MultiInputPolicy", env, policy_kwargs=policy_kwargs, verbose=1).learn(10000)

    # obs = env.reset()
    # env.render()
    # # print(env.observation_space)
    # # print(env.action_space)
    # # print(env.action_space.sample())
    #
    # for step in range(40):
    #     print("Step {}".format(step + 1))
    #     _, reward, done, info = env.step(env.action_space.sample())
    #     print('reward=', reward, 'done=', done)
    #     env.render()
    #     if done:
    #         print("Goal reached!", "reward=", reward)
    #         break


    # env.reset()
    # print(env.physics.named.data.xpos["box_1"])
    # print(env.physics.named.data.xquat["box_1"])
    # obs, _, _, info = env.step(np.array([0, 0, 0, 0, 1, -0.3]))




if __name__ == '__main__':
    config_class = TrainConfig()
    config = config_class.parse()

    config.sim_env = "/xmls/sand_ball_env.xml"
    config.task = "/reward"
    config.trained_models += config.task
    config.name = "sand_ball"
    config.suffix = "progress_reward_best_model"
    config.verbose = True

    if config.verbose:
        config_class.print_config(config)

    train(config)
