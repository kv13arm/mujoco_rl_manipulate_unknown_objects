from .base_config import BaseConfig


class TrainConfig(BaseConfig):
    def initialize(self, parser):
        parser = BaseConfig.initialize(self, parser)

        parser.add_argument('--task', type=str, default=None, help='adds training task name to saved model folder')
        parser.add_argument('--total_timesteps', type=int, default=int(0.5e6),
                            help='total number of timesteps to train the agent')
        parser.add_argument('--batch_size', type=int, default=256, help='size of minibatch')
        parser.add_argument('--eval_freq', type=int, default=2000, help='frequency of evaluation')
        parser.add_argument('--eval_episodes', type=int, default=3, help='number of episodes to evaluate')
        parser.add_argument('--buffer_size', type=int, default=int(0.5e6), help='size of replay buffer')

        return parser
