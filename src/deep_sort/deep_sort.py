from __future__ import annotations

import numpy as np

from typing import Any

from src.dataset.mot.mot_dataset_descriptor import MotDatasetDescriptor
from src.deep_sort import nn_matching, preprocessing
from src.deep_sort.detector.detections_provider import DetectionsProvider
from src.deep_sort.segmentation.segmentations_provider import SegmentationsProvider
from src.deep_sort.features_extractor.features_extractor import FeaturesExtractor
from src.deep_sort.track import Track
from src.deep_sort.tracker import Tracker
from src.utils.geometry.rect import Rect


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
        assert builder.dataset_descriptor is not None
        assert (builder.detections_provider is not None) \
               or (builder.segmentations_provider is not None)
        assert builder.features_extractor is not None

        self.__dataset_descriptor = builder.dataset_descriptor
        self.__detections_provider: DetectionsProvider = builder.detections_provider
        self.__segmentations_provider: SegmentationsProvider = builder.segmentations_provider
        self.__features_extractor: FeaturesExtractor = builder.features_extractor

        self.__detection_min_confidence = builder.detection_min_confidence
        self.__detection_min_height = builder.detection_min_height
        self.__detection_nms_max_overlap = builder.detection_nms_max_overlap

        self.__tracker = Tracker(metric=nn_matching.NearestNeighborDistanceMetric("cosine", 0.2),
                                 max_iou_distance=builder.tracking_max_iou_distance,
                                 max_age=builder.tracking_max_age,
                                 n_init=builder.tracking_n_init)

    def update(self,
               frame_id: int, image: np.ndarray) -> tuple[list[Track], list[Any]]:
        is_in_detections_mode = self.__detections_provider is not None

        filtered_detections_bboxes: list[Rect]
        first_stage_results: list[Any]

        if is_in_detections_mode:
            detections = self.__detections_provider.load_detections(image, frame_id)
            first_stage_results = detections

            # Filter detections with low confidence.
            detections = [detection for detection in detections if
                          detection.confidence >= self.__detection_min_confidence]
            bounding_boxes = np.array([list(detection.origin) for detection in detections])
            confidence_scores = np.array([d.confidence for d in detections])

            # Run non-maxima suppression.
            indices = preprocessing.non_max_suppression(bounding_boxes, self.__detection_nms_max_overlap,
                                                        confidence_scores)

            # We don't need confidence scores anymore and can disregard them after non-maxima suppression is done.
            filtered_detections_bboxes = [detections[i].origin for i in indices]
        else:
            # In segmentation mode.
            segmentations = self.__segmentations_provider.load_segmentations(image, frame_id)
            first_stage_results = segmentations

            # Filter segmentations with low confidence.
            segmentations = [segmentation for segmentation in segmentations if
                             segmentation.confidence >= self.__detection_min_confidence]
            bounding_boxes = np.array([list(segmentation.bbox) for segmentation in segmentations])
            confidence_scores = np.array([segmentation.confidence for segmentation in segmentations])

            # Run non-maxima suppression.
            indices = preprocessing.non_max_suppression(bounding_boxes, self.__detection_nms_max_overlap,
                                                        confidence_scores)

            # We don't need confidence scores anymore and can disregard them after non-maxima suppression is done.
            filtered_detections_bboxes = [segmentations[i].bbox for i in indices]

        # Run people identification on detected boxes.
        extracted_features = self.__features_extractor.extract(image, filtered_detections_bboxes)

        # Update tracker.
        self.__tracker.predict()
        self.__tracker.update(filtered_detections_bboxes, extracted_features)

        return self.__tracker.tracks, first_stage_results

    class Builder(object):
        def __init__(self,
                     dataset_descriptor: MotDatasetDescriptor):
            self.__dataset_descriptor = dataset_descriptor

            self.__detections_provider = None
            self.__segmentations_provider = None
            self.__features_extractor = None

            self.__detection_min_confidence = 0.8
            self.__detection_min_height = 0
            self.__detection_nms_max_overlap = 1.0

            self.__tracking_max_iou_distance: float = 0.7
            self.__tracking_max_age: int = 30
            self.__tracking_n_init: int = 3

        @property
        def dataset_descriptor(self) -> MotDatasetDescriptor:
            return self.__dataset_descriptor

        @property
        def detections_provider(self) -> DetectionsProvider:
            return self.__detections_provider

        @detections_provider.setter
        def detections_provider(self,
                                detections_provider: DetectionsProvider):
            assert self.__segmentations_provider is None
            self.__detections_provider = detections_provider

        @property
        def segmentations_provider(self) -> SegmentationsProvider:
            return self.__segmentations_provider

        @segmentations_provider.setter
        def segmentations_provider(self,
                                   segmentations_provider: SegmentationsProvider):
            assert self.__detections_provider is None
            self.__segmentations_provider = segmentations_provider

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
