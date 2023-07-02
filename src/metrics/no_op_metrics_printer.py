from src.metrics.metrics_printer import MetricsPrinter


class NoOpMetricsPrinter(MetricsPrinter):
    def add_sequence(self,
                     sequence: str,
                     metrics: dict[str, float]):
        # Empty on purpose.
        pass

    def print(self):
        # Empty on purpose.
        pass
