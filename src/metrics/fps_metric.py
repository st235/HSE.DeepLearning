from src.app.app import App
from src.dataset.mot.mot_ground_truth import MotGroundTruth
from src.metrics.metric import Metric
from src.utils.geometry.rect import Rect


class FPSMetric(Metric):

    KEY_METRIC_FPS = "FPS"

    def __init__(self,
                 app: App,
                 ground_truth: MotGroundTruth):
        super().__init__(ground_truth=ground_truth)

        self.__app = app

    def update_frame(self,
                     frame_id: int,
                     detections: dict[int, Rect]):
        # Empty on purpose to skip per frame evaluation.
        pass

    def evaluate(self) -> dict[str, float]:
        result_metrics: dict[str, float] = dict()

        fps_per_frames = self.__app.fps_records

        result_metrics[FPSMetric.KEY_METRIC_FPS] = sum(fps_per_frames) / len(fps_per_frames)

        return result_metrics
