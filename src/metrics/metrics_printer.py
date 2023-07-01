class MetricsPrinter(object):
    def __init__(self,
                 metrics_to_track: list[str]):
        assert len(metrics_to_track) > 0
        self.__metrics_to_track = metrics_to_track
        self.__sequences_metrics: dict[str, dict[str, float]] = dict()

    def add_sequence(self,
                     sequence: str,
                     metrics: dict[str, float]):
        assert self.__validate_metrics(metrics)
        assert sequence not in self.__sequences_metrics

        self.__sequences_metrics[sequence] = metrics

    def print(self):
        self.__print_header()
        self.__print_sequences()
        self.__print_combined()

    def __print_header(self):
        # Print table header.
        print(f"{'':20}|", end='')
        for metric_name in self.__metrics_to_track:
            print(f"{metric_name:10}|", end='')
        print()

    def __print_sequences(self):
        for sequence_name in sorted(self.__sequences_metrics.keys()):
            self.__print_sequence(sequence_name)

    def __print_sequence(self,
                         sequence_name: str):
        metrics = self.__sequences_metrics[sequence_name]
        print(f"{sequence_name:20}|", end='')
        for metric_name in self.__metrics_to_track:
            print(f"{metrics[metric_name]:10.5}|", end='')
        print()

    def __print_combined(self):
        metrics = dict()

        for metric_name in self.__metrics_to_track:
            for sequence_name in self.__sequences_metrics.keys():
                if metric_name not in metrics:
                    metrics[metric_name] = 0
                metrics[metric_name] += self.__sequences_metrics[sequence_name][metric_name]
            metrics[metric_name] /= len(self.__sequences_metrics)

        print(f"{'COMBINED':20}|", end='')
        for metric_name in self.__metrics_to_track:
            print(f"{metrics[metric_name]:10.5}|", end='')
        print()

    def __validate_metrics(self, metrics: dict) -> bool:
        matched = 0
        for metric_name in self.__metrics_to_track:
            if metric_name not in metrics:
                return False
            matched += 1
        return matched == len(self.__metrics_to_track)
