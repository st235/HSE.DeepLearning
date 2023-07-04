from __future__ import annotations

import numpy as np

from src.dataset.mot.mot_dataset_descriptor import MotDatasetDescriptor
from src.deep_sort import nn_matching, preprocessing
from src.deep_sort.detector.detections_provider import DetectionsProvider
from src.deep_sort.features_extractor.features_extractor import FeaturesExtractor
from src.deep_sort.track import Track
from src.deep_sort.tracker import Tracker


class DeepSort(object):
    """Deep Sort algorithm facade.

    Parameters
    ----------
    sequence_directory : str
        Path to the MOTChallenge sequence directory.
    detections_file : str
        Path to the detections file.
    output_file : str
        Path to the tracking output file. This file will contain the tracking
        results on completion.
    min_confidence : float
        Detection confidence threshold. Disregard all detections that have
        a confidence lower than this value.
    nms_max_overlap: float
        Maximum detection overlap (non-maxima suppression threshold).
    min_detection_height : int
        Detection height threshold. Disregard all detections that have
        a height lower than this value.
    max_cosine_distance : float
        Gating threshold for cosine distance metric (object appearance).
    nn_budget : Optional[int]
        Maximum size of the appearance descriptor gallery. If None, no budget
        is enforced.
    """

    def __init__(self,
                 builder: DeepSort.Builder):
        assert builder.database_descriptor is not None
        assert builder.detections_provider is not None
        assert builder.features_extractor is not None

        self.__dataset_descriptor = builder.database_descriptor
        self.__detections_provider = builder.detections_provider
        self.__features_extractor = builder.features_extractor

        self.__detection_min_confidence = builder.detection_min_confidence
        self.__detection_min_height = builder.detection_min_height
        self.__detection_nms_max_overlap = builder.detection_nms_max_overlap

        self.__tracker = Tracker(metric=nn_matching.NearestNeighborDistanceMetric("cosine", 0.2),
                                 max_iou_distance=builder.tracking_max_iou_distance,
                                 max_age=builder.tracking_max_age,
                                 n_init=builder.tracking_n_init)

    def update(self,
               frame_id: int, image: np.ndarray) -> list[Track]:
        detections = self.__detections_provider.load_detections(image, frame_id)
        # Filter detections with low confidence.
        detections = [detection for detection in detections if detection.confidence >= self.__detection_min_confidence]

        # Run non-maxima suppression.
        bounding_boxes = np.array([list(detection.origin) for detection in detections])
        confidence_scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(bounding_boxes, self.__detection_nms_max_overlap, confidence_scores)
        detections = [detections[i] for i in indices]

        # Run people identification on detected boxes.
        extracted_features = self.__features_extractor.extract(image, [detection.origin for detection in detections])

        # Update tracker.
        self.__tracker.predict()
        self.__tracker.update(detections, extracted_features)

        return self.__tracker.tracks

    class Builder(object):
        def __init__(self,
                     dataset_descriptor: MotDatasetDescriptor):
            self.__dataset_descriptor = dataset_descriptor

            self.__detections_provider = None
            self.__features_extractor = None

            self.__detection_min_confidence = 0.8
            self.__detection_min_height = 0
            self.__detection_nms_max_overlap = 1.0

            self.__tracking_max_iou_distance: float = 0.7
            self.__tracking_max_age: int = 30
            self.__tracking_n_init: int = 3

        @property
        def database_descriptor(self) -> MotDatasetDescriptor:
            return self.__dataset_descriptor

        @property
        def detections_provider(self) -> DetectionsProvider:
            return self.__detections_provider

        @detections_provider.setter
        def detections_provider(self,
                                detections_provider: DetectionsProvider):
            self.__detections_provider = detections_provider

        @property
        def features_extractor(self) -> FeaturesExtractor:
            return self.__features_extractor

        @features_extractor.setter
        def features_extractor(self,
                               features_extractor: FeaturesExtractor):
            self.__features_extractor = features_extractor

        @property
        def detection_min_confidence(self) -> float:
            return self.__detection_min_confidence

        @detection_min_confidence.setter
        def detection_min_confidence(self,
                                     detection_min_confidence: float):
            self.__detection_min_confidence = detection_min_confidence

        @property
        def detection_min_height(self) -> int:
            return self.__detection_min_height

        @detection_min_height.setter
        def detection_min_height(self,
                                 detection_min_height: int):
            self.__detection_min_height = detection_min_height

        @property
        def detection_nms_max_overlap(self) -> float:
            return self.__detection_nms_max_overlap

        @detection_nms_max_overlap.setter
        def detection_nms_max_overlap(self,
                                      detection_nms_max_overlap: float):
            self.__detection_nms_max_overlap = detection_nms_max_overlap

        @property
        def tracking_max_iou_distance(self) -> float:
            return self.__tracking_max_iou_distance

        @tracking_max_iou_distance.setter
        def tracking_max_iou_distance(self,
                                      tracking_max_iou_distance: float):
            self.__tracking_max_iou_distance = tracking_max_iou_distance

        @property
        def tracking_max_age(self) -> int:
            return self.__tracking_max_age

        @tracking_max_age.setter
        def tracking_max_age(self,
                             tracking_max_age: int):
            self.__tracking_max_age = tracking_max_age

        @property
        def tracking_n_init(self) -> float:
            return self.__tracking_n_init

        @tracking_n_init.setter
        def tracking_n_init(self,
                            tracking_n_init: float):
            self.__tracking_n_init = tracking_n_init

        def build(self) -> DeepSort:
            return DeepSort(builder=self)
