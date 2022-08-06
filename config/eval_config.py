from .base_config import BaseConfig


class EvalConfig(BaseConfig):
    def initialize(self, parser):
        parser = BaseConfig.initialize(self, parser)

        return parser
