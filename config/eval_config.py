from .base_config import BaseConfig


class EvalConfig(BaseConfig):
    def initialize(self, parser):
        parser = BaseConfig.initialize(self, parser)

        parser.add_argument('--eval_episodes', type=int, default=10, help='number of episodes to evaluate')

        return parser
