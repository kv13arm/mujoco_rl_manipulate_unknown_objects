import os
import gym
from config.train_config import TrainConfig
from simulation.environment.robot_env import RobotEnv
from models.feature_extractor import AugmentedNatureCNN
from models.callbacks import ProgressBarManager
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor


def train(config):
    env = RobotEnv(config)

    policy_kwargs = dict(features_extractor_class=AugmentedNatureCNN,
                         share_features_extractor=True)

    env = DummyVecEnv([lambda: Monitor(gym.make('Ant-Gripper-v0', config=config),
                                       os.path.join(model_save_dir, "log_file"))])
    test_env = DummyVecEnv([lambda: gym.make('gripper-env-v0', config=config)])

    # use deterministic actions for evaluation
    eval_callback = EvalCallback(test_env, best_model_save_path=model_save_dir,
                                 log_path=model_save_dir, eval_freq=2000, n_eval_episodes=2,
                                 deterministic=True, render=False)

    load_best_model = False
    if load_best_model:
        best_model = model_save_dir + 'best_model.zip'
        model = SAC.load(best_model, env, verbose=1)
        # print("Loaded model:", "gamma =", model.gamma)
    else:
        model = SAC("MultiInputPolicy", env, policy_kwargs=policy_kwargs, verbose=1)

    with ProgressBarManager(10000) as progress_callback:
        model.learn(10000, callback=[eval_callback, progress_callback])

    env.close()
    test_env.close()

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

    config.sim_env = "/xmls/sugar_cube_env.xml"
    config.task = "/reward"
    config.trained_models += config.task
    config.name = "sand_ball"
    config.suffix = "progress_reward_best_model"
    config.verbose = True

    if config.verbose:
        config_class.print_config(config)

    model_save_dir = config.trained_models + '/' + config.name

    train(config)
