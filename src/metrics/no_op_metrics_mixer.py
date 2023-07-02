from src.deep_sort.track import Track
from src.metrics.metrics_mixer import MetricsMixer


class NoOpMetricsMixer(MetricsMixer):
    def __init__(self):
        super().__init__(metrics=[])

    def update_frame(self,
                     frame_id: int,
                     tracks: list[Track]):
        # Empty on purpose.
        pass

    def evaluate(self) -> dict[str, float]:
        # Empty on purpose.
        return dict()
