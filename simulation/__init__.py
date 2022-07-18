from gym.envs.registration import register


register(
    id='gripper-env-v0',
    entry_point='simulation.env:RobotEnv',
    # max_episode_steps=1000,
)
