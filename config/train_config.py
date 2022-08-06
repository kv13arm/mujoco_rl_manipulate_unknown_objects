from .base_config import BaseConfig


class TrainConfig(BaseConfig):
    def initialize(self, parser):
        parser = BaseConfig.initialize(self, parser)

        parser.add_argument('--task', type=str, default=None, help='adds training task name to saved model folder')

        return parser
