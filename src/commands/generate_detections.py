import os
import errno
import argparse
import numpy as np
import cv2

from src.deep_sort.features_extractor.features_extractor import FeaturesExtractor
from src.deep_sort.features_extractor.tensorflow_v1_features_extractor import TensorflowV1FeaturesExtractor
from src.utils.geometry.rect import Rect


def _create_box_encoder(model_filename: str,
                        input_name: str = "images",
                        output_name: str = "features",
                        batch_size: int = 32) -> FeaturesExtractor:
    return TensorflowV1FeaturesExtractor(checkpoint_file=model_filename,
                                         input_name=input_name,
                                         output_name=output_name,
                                         batch_size=batch_size)


def _generate_detections(features_extractor: FeaturesExtractor,
                         mot_directory: str,
                         output_directory: str,
                         detection_directory: str = None):
    """Generate detections with features.

    Parameters
    ----------
    features_extractor : Callable[image, ndarray] -> ndarray
        The encoder function takes as input a BGR color image and a matrix of
        bounding boxes in format `(x, y, w, h)` and returns a matrix of
        corresponding feature vectors.
    mot_directory : str
        Path to the MOTChallenge directory (can be either train or test).
    output_directory
        Path to the output directory. Will be created if it does not exist.
    detection_directory
        Path to custom detections. The directory structure should be the default
        MOTChallenge structure: `[sequence]/det/det.txt`. If None, uses the
        standard MOTChallenge detections.

    """
    if detection_directory is None:
        detection_directory = mot_directory
    try:
        os.makedirs(output_directory)
    except OSError as exception:
        if exception.errno == errno.EEXIST and os.path.isdir(output_directory):
            pass
        else:
            raise ValueError(
                "Failed to created output directory '%s'" % output_directory)

    for sequence in os.listdir(mot_directory):
        print("Processing %s" % sequence)
        sequence_dir = os.path.join(mot_directory, sequence)

        if not os.path.isdir(sequence_dir):
            # Filter out hidden files, like, system ones.
            continue

        image_dir = os.path.join(sequence_dir, "img1")
        image_filenames = {
            int(os.path.splitext(f)[0]): os.path.join(image_dir, f)
            for f in os.listdir(image_dir)}

        detection_file = os.path.join(
            detection_directory, sequence, "det/det.txt")
        detections_in = np.loadtxt(detection_file, delimiter=',')
        detections_out = []

        frame_indices = detections_in[:, 0].astype(np.int32)
        min_frame_idx = frame_indices.astype(np.int32).min()
        max_frame_idx = frame_indices.astype(np.int32).max()
        for frame_idx in range(min_frame_idx, max_frame_idx + 1):
            print("Frame %05d/%05d" % (frame_idx, max_frame_idx))
            mask = frame_indices == frame_idx
            rows = detections_in[mask]

            if frame_idx not in image_filenames:
                print("WARNING could not find image for frame %d" % frame_idx)
                continue
            bgr_image = cv2.imread(image_filenames[frame_idx], cv2.IMREAD_COLOR)

            bbox_rects = []

            for row_index in range(rows.shape[0]):
                row = rows[row_index, :]
                bbox_rects.append(Rect.from_tlwh(row[2:6]))

            features = features_extractor.extract(bgr_image, bbox_rects)
            detections_out += [np.r_[(row, feature)] for row, feature
                               in zip(rows, features)]

        output_filename = os.path.join(output_directory, "%s.npy" % sequence)
        np.save(
            output_filename, np.asarray(detections_out), allow_pickle=False)


def _parse_args():
    """Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Re-ID feature extractor")
    parser.add_argument(
        "--model",
        default="resources/networks/mars-small128.pb",
        help="Path to freezed inference graph protobuf.")
    parser.add_argument(
        "--mot_dir", help="Path to MOTChallenge directory (train or test)",
        required=True)
    parser.add_argument(
        "--detection_dir", help="Path to custom detections. Defaults to "
        "standard MOT detections Directory structure should be the default "
        "MOTChallenge structure: [sequence]/det/det.txt", default=None)
    parser.add_argument(
        "--output_dir", help="Output directory. Will be created if it does not"
        " exist.", default="detections")
    return parser.parse_args()


def main():
    args = _parse_args()
    encoder = _create_box_encoder(args.model, batch_size=32)
    _generate_detections(encoder, args.mot_dir, args.output_dir,
                         args.detection_dir)


if __name__ == "__main__":
    main()
