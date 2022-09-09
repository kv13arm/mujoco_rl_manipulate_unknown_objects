import os
from pathlib import Path
from simulation.environment.robot_env import RobotEnv
from config.train_config import TrainConfig
from models.feature_extractor import AugmentedNatureCNN
from models.callbacks import ProgressBarManager, SaveOnBestTrainingRewardCallback
from stable_baselines3 import SAC, HerReplayBuffer
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from stable_baselines3.common.monitor import Monitor


def train(config):
    model_save_dir = Path(__file__).resolve().parent.as_posix() + config.trained_models + '/' + config.name
    os.makedirs(model_save_dir, exist_ok=True)

    make_env = RobotEnv(config)
    policy_kwargs = dict(features_extractor_class=AugmentedNatureCNN,
                         share_features_extractor=True,
                         net_arch=[256, 256])

    env = DummyVecEnv([lambda: Monitor(make_env,
                                       os.path.join(model_save_dir, "log_file"))])

    env = VecVideoRecorder(env,
                           video_folder=model_save_dir + "/videos",
                           record_video_trigger=lambda x: x % config.record_freq == 0,
                           video_length=config.vid_length,
                           name_prefix=f"{config.name}")

    test_env = DummyVecEnv([lambda: make_env])

    # use deterministic actions for evaluation
    eval_callback = EvalCallback(test_env,
                                 best_model_save_path=model_save_dir,
                                 log_path=model_save_dir,
                                 eval_freq=config.eval_freq,
                                 n_eval_episodes=config.eval_episodes,
                                 deterministic=True,
                                 render=config.render_eval)

    # save best model based on mean reward
    save_callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=model_save_dir)

    load_best_model = False
    if load_best_model:
        best_model = model_save_dir + '/best_model.zip'
        model = SAC.load(best_model,
                         env,
                         verbose=1)

        with ProgressBarManager(config.total_timesteps) as progress_callback:
            model.learn(config.total_timesteps,
                        callback=[eval_callback, progress_callback, save_callback],
                        reset_num_timesteps=False)

    else:
        if config.her_buffer:
            goal_selection_strategy = 'future'

            model = SAC("MultiInputPolicy",
                        env,
                        replay_buffer_class=HerReplayBuffer,
                        # Parameters for HER
                        replay_buffer_kwargs=dict(
                            n_sampled_goal=4,
                            goal_selection_strategy=goal_selection_strategy,
                            online_sampling=True,
                            max_episode_length=config.time_horizon),
                        learning_starts=config.time_horizon,
                        policy_kwargs=policy_kwargs,
                        buffer_size=config.buffer_size,
                        batch_size=config.batch_size,
                        verbose=1,
                        tensorboard_log=model_save_dir)
            print("Using HER buffer")

            with ProgressBarManager(config.total_timesteps) as progress_callback:
                model.learn(config.total_timesteps,
                            callback=[eval_callback, progress_callback, save_callback])
        else:
            model = SAC("MultiInputPolicy",
                        env,
                        policy_kwargs=policy_kwargs,
                        buffer_size=config.buffer_size,
                        batch_size=config.batch_size,
                        verbose=1,
                        tensorboard_log=model_save_dir)

            with ProgressBarManager(config.total_timesteps) as progress_callback:
                model.learn(config.total_timesteps,
                            callback=[eval_callback, progress_callback, save_callback])

    # model.save(model_save_dir)

    env.close()
    test_env.close()


if __name__ == '__main__':
    config_class = TrainConfig()
    config = config_class.parse()
    config.trained_models = f"{config.trained_models}/dir{config.direction}_{config.task}"

    # config.sim_env = "/xmls/sand_ball_env.xml"
    # config.task = "reward"
    # config.name = "sand_ball"
    # config.suffix = "progress_reward_best_model"
    # config.verbose = True
    # config.render_eval = False

    if config.verbose:
        config_class.print_config(config)

    train(config)
