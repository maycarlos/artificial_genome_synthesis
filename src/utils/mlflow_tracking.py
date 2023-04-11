import contextlib
from pathlib import Path

import mlflow


class MLFlowTracker:
    def __init__(self, tracking_path: Path) -> None:
        self.tracking_folder = tracking_path

    @contextlib.contextmanager
    def start_run(self):
        pass

    def set_tracker(self):
        mlflow.start_run()

        mlflow.stop
