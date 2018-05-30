from typing import Dict, Text

from rasa_nlu.config import RasaNLUModelConfig
from rasa_nlu.train import do_train_in_worker


class InplaceTraining(object):
    @classmethod
    def train(cls, config, data, project_dir_path,
              project=None, fixed_model_name=None,
              storage=None):
        # type: (Dict, Text, str, str, str, str) -> None
        cfg = RasaNLUModelConfig(config)

        do_train_in_worker(
            cfg,
            data,
            path=project_dir_path,
            project=project,
            fixed_model_name=fixed_model_name,
            storage=storage,
            component_builder=None
        )


inplace_train = InplaceTraining.train
