from __future__ import division, print_function, absolute_import

import os.path
from typing import Optional

import numpy as np

from dependencies.nanodet_binaries.nanodet_model import NanoDetPaths
from src.app.app import App
from src.app.visualization import Visualization
from src.dataset.mot.mot_dataset_descriptor import MotDatasetDescriptor
from src.deep_sort.deep_sort import DeepSort
from src.deep_sort.detector.detections_provider import DetectionsProvider
from src.deep_sort.detector.file_detections_provider import FileDetectionsProvider
from src.deep_sort.detector.ground_truth_detections_provider import GroundTruthDetectionsProvider
from src.deep_sort.detector.hog_detections_provider import HogDetectionsProvider
from src.deep_sort.detector.mmdetection_detections_provider import MmdetectionDetectionsProvider
from src.deep_sort.detector.nanodet_detections_provider import NanodetDetectionsProvider
from src.deep_sort.detector.yolov5_detections_provider import YoloV5DetectionsProvider
from src.deep_sort.detector.yolov8_detections_provider import YoloV8DetectionsProvider
from src.deep_sort.features_extractor.features_extractor import FeaturesExtractor
from src.deep_sort.features_extractor.tensorflow_v1_features_extractor import TensorflowV1FeaturesExtractor
from src.deep_sort.features_extractor.torchreid_features_extractor import TorchReidFeaturesExtractor
from src.metrics.metrics_mixer import MetricsMixer
from src.metrics.metrics_printer import MetricsPrinter
from src.metrics.no_op_metrics_mixer import NoOpMetricsMixer
from src.metrics.no_op_metrics_printer import NoOpMetricsPrinter
from src.metrics.std_metrics_printer import StdMetricsPrinter


def run(sequence_directories: list[str],
        detector: str,
        features_extractor: str,
        output_file: str,
        min_confidence: float,
        nms_max_overlap: int,
        min_detection_height: int,
        max_cosine_distance: float,
        nn_budget: int,
        max_iou_distance: float,
        max_age: int,
        n_init: int,
        metrics_to_track: Optional[list[str]],
        extra_param: Optional[str]):
    metrics_printer: MetricsPrinter
    if metrics_to_track is not None:
        metrics_printer = StdMetricsPrinter(metrics_to_track=metrics_to_track)
    else:
        metrics_printer = NoOpMetricsPrinter()

    for sequence_path in sequence_directories:
        sequence_name = os.path.basename(os.path.normpath(sequence_path))
        print(f"Running {sequence_name} sequence")

        metrics = __run_sequence(sequence_path,
                                 detector,
                                 features_extractor,
                                 output_file,
                                 min_confidence,
                                 nms_max_overlap,
                                 min_detection_height,
                                 max_cosine_distance,
                                 nn_budget,
                                 max_iou_distance,
                                 max_age,
                                 n_init,
                                 metrics_to_track,
                                 extra_param)

        metrics_printer.add_sequence(sequence_name, metrics)

    print()
    metrics_printer.print()


def __run_sequence(sequence_directory: str,
                   detector: str,
                   features_extractor: str,
                   output_file: str,
                   min_confidence: float,
                   nms_max_overlap: int,
                   min_detection_height: int,
                   max_cosine_distance: float,
                   nn_budget: int,
                   max_iou_distance: float,
                   max_age: int,
                   n_init: int,
                   metrics_to_track: Optional[list[str]],
                   extra_param: Optional[str]) -> dict[str, float]:
    """Run multi-target tracker on a particular sequence.

    Parameters
    ----------
    sequence_directory: str
        Path to the MOTChallenge sequence directory.
    output_file: str
        Path to the tracking output file. This file will contain the tracking
        results on completion.
    min_confidence: float
        Detection confidence threshold. Disregard all detections that have
        a confidence lower than this value.
    nms_max_overlap: float
        Maximum detection overlap (non-maxima suppression threshold).
    min_detection_height: int
        Detection height threshold. Disregard all detections that have
        a height lower than this value.
    max_cosine_distance: float
        Gating threshold for cosine distance metric (object appearance).
    nn_budget: Optional[int]
        Maximum size of the appearance descriptor gallery. If None, no budget
        is enforced.
    """

    dataset_descriptor = MotDatasetDescriptor.load(sequence_directory)

    app = App(dataset_descriptor)

    deep_sort_builder = DeepSort.Builder(dataset_descriptor=dataset_descriptor)
    deep_sort_builder.detections_provider = __create_detector_by_name(detector, dataset_descriptor, extra_param)
    deep_sort_builder.features_extractor = __create_features_extractor_by_name(features_extractor)

    deep_sort_builder.detection_min_confidence = min_confidence
    deep_sort_builder.detection_nms_max_overlap = nms_max_overlap
    deep_sort_builder.detection_min_height = min_detection_height

    deep_sort_builder.tracking_max_iou_distance = max_iou_distance
    deep_sort_builder.tracking_max_age = max_age
    deep_sort_builder.tracking_n_init = n_init

    deep_sort = deep_sort_builder.build()

    detections_hypotheses: list[list[float]] = []

    metrics_mixer: MetricsMixer
    if metrics_to_track is not None:
        metrics_mixer = MetricsMixer.create_for_metrics(ground_truth=dataset_descriptor.ground_truth,
                                                        metrics_to_track=set(metrics_to_track))
    else:
        metrics_mixer = NoOpMetricsMixer()

    def frame_callback(frame_id: int, image: np.ndarray, visualisation: Visualization):
        tracks = deep_sort.update(frame_id, image)

        visualisation.draw_trackers(tracks)

        metrics_mixer.update_frame(frame_id, tracks)

        # Store results.
        for track in tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = list(track.bounding_box)
            detections_hypotheses.append([frame_id, track.track_id, bbox[0], bbox[1], bbox[2], bbox[3]])

    # Run the app.
    app.display_fps()
    app.start(frame_callback)

    # Store results.
    f = open(output_file, 'w')
    for row in detections_hypotheses:
        print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (
            row[0], row[1], row[2], row[3], row[4], row[5]), file=f)

    return metrics_mixer.evaluate()


def get_supported_detectors() -> set[str]:
    """Returns supported detectors.
    """
    return {'det', 'gt', 'hog',
            'mmdet_darknet', 'mmdet_mobilenetv2', 'mmdet_yoloxt', 'mmdet_yoloxs', 'mmdet_yoloxl',
            'nanodet_legacy', 'nanodet_plusm320', 'nanodet_plusm15x320', 'nanodet_plusm416', 'nanodet_plusm15x416',
            'yolov5n', 'yolov5n6', 'yolov5s', 'yolov5m', 'yolov5l',
            'yolov8n', 'yolov8s', 'yolov8m', 'yolov8l'}


def __create_detector_by_name(detector: str,
                              dataset_descriptor: MotDatasetDescriptor,
                              extra_param: Optional[str]) -> DetectionsProvider:
    if detector == 'det':
        return FileDetectionsProvider(detections=dataset_descriptor.detections)

    if detector == 'gt':
        return GroundTruthDetectionsProvider(ground_truth=dataset_descriptor.ground_truth)

    if detector == 'hog':
        return HogDetectionsProvider()

    if detector == 'nanodet_legacy':
        return NanodetDetectionsProvider(paths=NanoDetPaths.LegacyM)

    if detector == 'nanodet_plusm320':
        return NanodetDetectionsProvider(paths=NanoDetPaths.PlusM320)

    if detector == 'nanodet_plusm15x320':
        return NanodetDetectionsProvider(paths=NanoDetPaths.PlusM15X320)

    if detector == 'nanodet_plusm416':
        return NanodetDetectionsProvider(paths=NanoDetPaths.PlusM416)

    if detector == 'nanodet_plusm15x416':
        return NanodetDetectionsProvider(paths=NanoDetPaths.PlusM15X416)

    if detector == 'mmdet_darknet':
        return MmdetectionDetectionsProvider(config=MmdetectionDetectionsProvider.Config.DarkNet53608,
                                             model=extra_param)

    if detector == 'mmdet_mobilenetv2':
        return MmdetectionDetectionsProvider(config=MmdetectionDetectionsProvider.Config.MobileNetV2,
                                             model=extra_param)

    if detector == 'mmdet_yoloxt':
        return MmdetectionDetectionsProvider(config=MmdetectionDetectionsProvider.Config.YoloXT,
                                             model=extra_param)

    if detector == 'mmdet_yoloxs':
        return MmdetectionDetectionsProvider(config=MmdetectionDetectionsProvider.Config.YoloXS,
                                             model=extra_param)

    if detector == 'mmdet_yoloxl':
        return MmdetectionDetectionsProvider(config=MmdetectionDetectionsProvider.Config.YoloXL,
                                             model=extra_param)

    if detector == 'yolov5n':
        return YoloV5DetectionsProvider(checkpoint=YoloV5DetectionsProvider.Checkpoint.NANO)

    if detector == 'yolov5n6':
        return YoloV5DetectionsProvider(checkpoint=YoloV5DetectionsProvider.Checkpoint.NANO6)

    if detector == 'yolov5s':
        return YoloV5DetectionsProvider(checkpoint=YoloV5DetectionsProvider.Checkpoint.SMALL)

    if detector == 'yolov5m':
        return YoloV5DetectionsProvider(checkpoint=YoloV5DetectionsProvider.Checkpoint.MEDIUM)

    if detector == 'yolov5l':
        return YoloV5DetectionsProvider(checkpoint=YoloV5DetectionsProvider.Checkpoint.LARGE)

    if detector == 'yolov8n':
        return YoloV8DetectionsProvider(checkpoint=YoloV8DetectionsProvider.Checkpoint.NANO)

    if detector == 'yolov8s':
        return YoloV8DetectionsProvider(checkpoint=YoloV8DetectionsProvider.Checkpoint.SMALL)

    if detector == 'yolov8m':
        return YoloV8DetectionsProvider(checkpoint=YoloV8DetectionsProvider.Checkpoint.MEDIUM)

    if detector == 'yolov8l':
        return YoloV8DetectionsProvider(checkpoint=YoloV8DetectionsProvider.Checkpoint.LARGE)

    raise Exception(f"Unknown detector type {detector}")


def get_supported_features_extractor() -> set[str]:
    """Returns supported features extractor.
    """
    return {'ftv1',
            'torchreid_shufflenet', 'torchreid_mobilenet', 'torchreid_mobilenet14x',
            'torchreid_mlfn', 'torchreid_osnet', 'torchreid_osnet075', 'torchreid_osnetibn',
            'torchreid_osnetain', 'torchreid_osnetain075'}


def __create_features_extractor_by_name(features_extractor: str) -> FeaturesExtractor:
    if features_extractor == 'tfv1':
        return TensorflowV1FeaturesExtractor.create_default()

    if features_extractor == 'torchreid_shufflenet':
        return TorchReidFeaturesExtractor(model=TorchReidFeaturesExtractor.Model.Shufflenet)

    if features_extractor == 'torchreid_mobilenet':
        return TorchReidFeaturesExtractor(model=TorchReidFeaturesExtractor.Model.Mobilenet)

    if features_extractor == 'torchreid_mobilenet14x':
        return TorchReidFeaturesExtractor(model=TorchReidFeaturesExtractor.Model.Mobilenet14x)

    if features_extractor == 'torchreid_mlfn':
        return TorchReidFeaturesExtractor(model=TorchReidFeaturesExtractor.Model.Mlfn)

    if features_extractor == 'torchreid_osnet':
        return TorchReidFeaturesExtractor(model=TorchReidFeaturesExtractor.Model.Osnet)

    if features_extractor == 'torchreid_osnet075':
        return TorchReidFeaturesExtractor(model=TorchReidFeaturesExtractor.Model.Osnet075)

    if features_extractor == 'torchreid_osnetibn':
        return TorchReidFeaturesExtractor(model=TorchReidFeaturesExtractor.Model.OsnetIbn)

    if features_extractor == 'torchreid_osnetain':
        return TorchReidFeaturesExtractor(model=TorchReidFeaturesExtractor.Model.OsnetAin)

    if features_extractor == 'torchreid_osnetain075':
        return TorchReidFeaturesExtractor(model=TorchReidFeaturesExtractor.Model.OsnetAin075)

    raise Exception(f"Unknown features extractor {features_extractor}")
