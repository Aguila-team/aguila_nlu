from typing import Any

from rasa_nlu.project_manager import ProjectManager
from rasa_nlu.components import ComponentBuilder


class InplaceParsing(object):
    def __init__(self, project_dir, remote_storage,
                 query_logger=None, component_builder=None):
        # type: (str, str, Any, ComponentBuilder) -> None
        self.query_logger = query_logger

        self.project_loader = ProjectManager(
            project_dir=project_dir,
            remote_storage=remote_storage,
            component_builder=component_builder
        )

    def parse(self, text, project, model, time=None):
        project = self.project_loader.load_project(project)

        response, used_model = project.parse(text, time, model)

        if self.query_logger:
            self.query_logger.info('', user_input=response, project=project,
                                   model=used_model)
        return response
