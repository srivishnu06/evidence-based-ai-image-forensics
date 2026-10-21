from pathlib import Path


class NullTracker:
    enabled = False
    task = None

    def connect_parameters(self, *args, **kwargs): return None
    def report_scalar(self, *args, **kwargs): return None
    def report_single_value(self, *args, **kwargs): return None
    def report_text(self, *args, **kwargs): return None
    def upload_model(self, *args, **kwargs): return None
    def upload_artifact(self, *args, **kwargs): return None
    def close(self): return None


class ClearMLTracker:
    """Thin wrapper so ClearML remains optional in training/evaluation scripts."""

    enabled = True

    def __init__(self, project_name, task_name, task_type="training", output_uri=None, tags=None):
        from clearml import Task

        task_type_map = {
            "training": Task.TaskTypes.training,
            "testing": Task.TaskTypes.testing,
            "inference": Task.TaskTypes.inference,
            "data_processing": Task.TaskTypes.data_processing,
        }
        self.task = Task.init(
            project_name=project_name,
            task_name=task_name,
            task_type=task_type_map.get(task_type, Task.TaskTypes.training),
            tags=tags or [],
            output_uri=output_uri,
            reuse_last_task_id=False,
        )
        self.logger = self.task.get_logger()

    def connect_parameters(self, params, name="Parameters"):
        self.task.connect(params, name=name)

    def report_scalar(self, title, series, value, iteration):
        self.logger.report_scalar(title=title, series=series, value=float(value), iteration=int(iteration))

    def report_single_value(self, name, value):
        self.logger.report_single_value(name=name, value=float(value))

    def report_text(self, text):
        self.logger.report_text(str(text))

    def upload_model(self, model_path, model_name=None):
        path = Path(model_path)
        if path.exists():
            self.task.update_output_model(
                model_path=str(path),
                name=model_name or path.stem,
                auto_delete_file=False,
            )

    def upload_artifact(self, name, artifact):
        self.task.upload_artifact(name=name, artifact_object=artifact)

    def close(self):
        if self.task is not None:
            self.task.close()


def create_tracker(enabled, project_name, task_name, task_type="training", output_uri=None, tags=None):
    if not enabled:
        return NullTracker()
    return ClearMLTracker(
        project_name=project_name,
        task_name=task_name,
        task_type=task_type,
        output_uri=output_uri,
        tags=tags,
    )
