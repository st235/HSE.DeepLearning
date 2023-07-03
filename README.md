# Deep SORT

## Introduction

This repository contains code for *Simple Online and Realtime Tracking with a Deep Association Metric* (Deep SORT).

It extends [the original DeepSORT](https://github.com/nwojke/deep_sort) algorithm to integrate
improved detections and Deep REID mechanisms.

## Data

The project uses [Multiple Object Tracking Benchmark](https://motchallenge.net/) for evaluation.

The project is using the following videos:
- KITTI-17
- MOT16-09
- MOT16-11
- PETS09-S2L1
- TUD-Campus
- TUD-Stadtmitte

MOT is published under [ Creative Commons Attribution-NonCommercial-ShareAlike 3.0 License](https://creativecommons.org/licenses/by-nc-sa/3.0/).

| MOT16-11                                   | MOT16-06                                   | MOT16-13                                   | MOT16-01                                   | MOT16-14                                   |
|--------------------------------------------|--------------------------------------------|--------------------------------------------|--------------------------------------------|--------------------------------------------|
| ![MOT16-11](./resources/dataset_mot_1.jpg) | ![MOT16-06](./resources/dataset_mot_2.jpg) | ![MOT16-13](./resources/dataset_mot_3.jpg) | ![MOT16-01](./resources/dataset_mot_4.jpg) | ![MOT16-14](./resources/dataset_mot_5.jpg) |

## Dependencies

The code is compatible with Python 3.9+. 
The following dependencies are needed to run the tracker:

* NumPy
* sklearn
* OpenCV
* TensorFlow
* Pandas
* Torchvision

The full list of requirements can be found in [requirements.txt](./requirements.txt).
Moreover, the project depends on other projects via [git sumbodules system](https://git-scm.com/book/en/v2/Git-Tools-Submodules).
You can find all declared submodules in [.gitmodules](./.gitmodules) or under [the dependencies folder](./dependencies).

## Running

### Setup

First of all, it is necessary to **install** the project.

To do so just run the following commands:

```bash
pip install -r requirements.txt
python setup.py develop
```

We are good to go and run the project! Yay 🎉

### Commands

There are 2 modes to run the app:
- **Ground truth visualisation** that does not start the tracker but visualise ground truth from `gt` folder
- **Tracker** runs the algorithm and this is _the main part of the project_

This is an example of how to run `ground truth visualisation`.

```bash
deep-sort ground-truth ./data/sequences/MOT16-11
```

You will see something similar to the image below:

![Ground truth visualisation](./resources/ground_truth_visualisation.png)

The following example runs the tracker:

```bash
deep-sort run ./data/sequences -e HOTA DetA AssA F1 Precision Recall 
```

In this case you will see something similar to the next image:

![Run DeepSORT](./resources/run_deep_sort.png)

Check `deep-sort -h` for an overview of available options.

## Overview of source files

### Detection

![Detections Provider and Detection diagram](./resources/detections_provider.png)

Detections logic is located in [the detector folder](./src/deep_sort/detector) under **deep_sort**.
There are 2 central abstractions in the project: _Detection_ and _DetectionsProvider_.

_DetectionsProvider_ helps to abstract a specific detections logic under a unified interface.
Every provider should return _a lift of Detections_ in a given sequence frame.
A frame is described by the image matrix (np.ndarray) and a frame id.
The provided list of detections helps to identify an origin of every detection,
i.e. where the particular detection is located in the frame, and a confidence score.

There are a few classes implementing __DetectionsProvider__.

#### Original FileDetectionsProvider

![FileDetectionsProvider](./resources/file_detections_provider.png

_FileDetectionsProvider_ is the logic that **was used in the original DeepSORT** version of the project.
It reads the 


#### NanodetDetectionsProvider

[NanodetDetectionsProvider](src/deep_sort/detector/nanodet_detections_provider.py)
is using [Nanodet](https://github.com/RangiLyu/nanodet) model.

Unfortunately, this library does not work on MacOS, therefore `Nanodet` used in this project is a _patched_ version.

This is the patch that was sent to the authors of `Nanodet`:
- [#516 Guard CUDA calls with an explicit check](https://github.com/RangiLyu/nanodet/pull/516)


### REID

#### Features extraction

![Features extractor](./resources/features_extractor.png)

Features extractor is the main abstraction for converting a detection area into a feature vector.

The API of the class looks like:

| Method                                                          | Description                                                                                                                                 |
|-----------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------|
| **extract(image: np.ndarray, boxes: list[Rect]) -> np.ndarray** | it accepts original image and found detections, and returns a list of features vectors. Feature vectors go in the same order as detections. |

#### Tensorflow V1

![TFV1 Features extractor](./resources/tfv1_features_extractor.png)

[TensorflowV1FeaturesExtractor](./src/deep_sort/features_extractor/tensorflow_v1_features_extractor.py) provides
a tensorflow model to extract feature vectors from detections.

### Misc

#### Geometry, Rect

![Rect](./resources/rect.png)

Rect represents a rectangular area and helps to deal with their geometry. It provides a few helpful methods:

| Method                                                                  | Description                                                                                                                                                    |
|-------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **width: float**                                                        | returns width of the rect                                                                                                                                      |
| **height: float**                                                       | returns height of the rect                                                                                                                                     |
| **top: float**                                                          | returns first horizontal pixels position (aka top edge) in the original image                                                                                  |
| **left: float**                                                         | returns first vertical pixels position (aka left edge) in the original image                                                                                   |
| **right: float**                                                        | returns last vertical pixels position (aka right edge) in the original image                                                                                   |
| **bottom: float**                                                       | returns last horizontal pixels position (aka bottom edge) in the original image                                                                                |
| **center_x: float**                                                     | returns horizontal central position                                                                                                                            |
| **center_y: float**                                                     | returns vertical central position                                                                                                                              |
| **aspect_ratio: float**                                                 | returns ratio of width to height, i.e. `width / height`                                                                                                        |
| **area: float**                                                         | returns rectangle area, i.e. `width * height`                                                                                                                  |
| **inset(left: float, top: float, right: float, bottom: float) -> Rect** | adds paddings to the current rect and returns a new rect with new paddings                                                                                     |
| **check_if_intersects(that: Rect) -> bool**                             | checks if 2 rectangles are intersecting                                                                                                                        | 
| **iou(hat: Rect) -> float**                                             | calculates intersection over union, the return value is always within **[0, 1]**                                                                               |
| **resize(target_width: float, target_height: float) -> Rect**           | scales current rect and returns a new one with the same aspect ratio as target_width over target_height                                                        |
| **clip(that: Rect) -> Rect**                                            | clips the other rect by the bounding boxes of the current rect or raises exception if the other box is completely outside the bounding box of the current rect |

Rect fixes an issue within the original [`deep sort`](https://github.com/nwojke/deep_sort). Bottom right corners of
the bounding boxes are calculated incorrectly which _may affect detection and metrics calculation quality_.

This is the Pull Request that fixes the issue:
- [#314 Fix bbox bottom right corner calculation](https://github.com/nwojke/deep_sort/pull/314)


## Metric

### HOTA

Metric used in this project is called _Higher Order Tracking Accuracy_ (aka HOTA).

The metric consists of a few sub metrics:

#### 1. Localization

Finds the spatial alignment between predicted detections and ground truth detections.
[_IOU_](https://en.wikipedia.org/wiki/Jaccard_index) is used to find localisation between **one** detection and **one**
ground truth object.

![Intersection Over Union picture](./resources/iou.png)

Overall Localization Accuracy (LocA) is calculated over all pairs across the **entire dataset**.

```math
LocA = \frac{1}{|TP|} \sum_{c \in TP} Loc-IOU(c)
```

In the codebase iou implemented in [iou utils](./src/utils/geometry/iou_utils.py) and [Rect](./src/utils/geometry/rect.py)

#### 2. Detection

Detection measures the alignment between all predicted detections and ground truth detections.
We rely on _localisation_ results to find the overlap between predictions and ground truth. To break the tie
when there are more than one prediction intersect with a ground truth _Hungarian algorithm_ (aka assignment problem algorithm) is used.

Implementation of the assignment algorithm is used from [scipy.linear_sum_assignment](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html#scipy-optimize-linear-sum-assignment),
in the latest release of the library _Hungarian algorithm_ has been replaced with _Jonker-Volgenant algorithm_.

After running the algorithm we end up with matched and unmatched elements. We can divide them into 3 groups:
True Positives (intersection between the two sets of detections), False Positives (predicted detections that don’t match),
and False Negatives (ground-truth detections that don’t match).

Overall Detection Accuracy (aka DetA) is calcuated by using the count of TPs, FNs and FPs over the whole dataset.

```math
DetA=\frac{|TP|}{|TP|+|FP|+|FN|}
```

#### 3. Association

Show how well the tracking links detections **over time** into the **same identities**.

The intersection between two tracks can be found in a similar way as during the detection step, but with
a little difference: True Positive Associations (number of True Positive matches between the two tracks), 
False Positive Associations (any remaining detections in the predicted track which are either matched to other ground-truth tracks or none at all),
and False Negative Associations (any remaining detections in the ground-truth track).

See visual example of the definitions TPA, FNA and FPA:

![Example of Association Metrics](./resources/assa.png)

Overall Association Accuracy (aka AssA) is calculated for **every** True Positive pair **across the entire dataset**.

```math
AssA=\frac{1}{|TP|} \sum_{c \in TP} \frac{|TPA(c)|}{|TPA(c)|+|FPA(c)|+|FNA(c)|}
```

#### Gathering sub-metrics together

Detection and association were defined using a _Hungarian matching_ based on a certain _Loc-IoU threshold_ (_α_). 
Since they both depend on the quality of localisation we calculate them over a range of different _α_ thresholds.

_HOTA_ for specific alpha a can be calculated as:

```math
HOTA_{\alpha}=\sqrt{DetA_{\alpha}*AssA_{\alpha}}
```

Overall _HOTA_ is a **discrete integral** over different alphas:

```math
HOTA=\int_{0 < \alpha \le 1} HOTA_{\alpha} \thickapprox \sum_{\alpha=0.05,\space \alpha += 0.05}^{0.95} HOTA_{\alpha}
```

#### Implementation

- [HotaMetric](./src/metrics/hota_metric.py)


#### References

- [How to evaluate tracking with the HOTA metricsPermalink](https://autonomousvision.github.io/hota-metrics/)
- [HOTA: A Higher Order Metric for Evaluating Multi-object Tracking](https://link.springer.com/article/10.1007/s11263-020-01375-2)
- [TrackEval](https://github.com/JonathonLuiten/TrackEval/blob/master/trackeval/metrics/hota.py)

### Confusion Matrix Metrics

This is implementation of metrics used primarily for detection assessments

#### Precision

```math
Precision=\frac{|TP|}{|TP|+|FP|}
```

#### Recall

```math
Recall=\frac{|TP|}{|TP|+|FN|}
```

#### F1

```math
F1=2 * \frac{Precision*Recall}{Precision+Recall}
```

#### Implementation

- [ConfusionMatrixMetric](./src/metrics/confusion_matrix_metric.py)

### Usage

You can enable evaluation in [the main script](./src/commands/deep_sort.py) using the command below:

```bash
deep-sort run ./data/sequences --eval HOTA DetA AssA F1 Recall Precision
```

Supported metrics are:
- HOTA
- DetA (calculated as a part of HOTA)
- AssA (calculated as a part of HOTA)
- F1
- Recall
- Precision

You will see a table similar to the table below:

```text
                    |HOTA      |DetA      |AssA      |F1        |Recall    |Precision |
KITTI-17            |   0.39411|    0.4061|   0.38431|   0.52665|   0.37511|   0.88362|
MOT16-09            |   0.30053|   0.27235|   0.33223|   0.41727|   0.26783|   0.94404|
MOT16-11            |   0.40245|   0.33696|   0.48102|    0.4623|   0.30172|   0.98828|
PETS09-S2L1         |    0.4469|   0.48727|   0.41129|   0.57651|   0.46904|   0.74787|
TUD-Campus          |    0.4007|   0.44327|   0.36568|   0.53687|   0.39358|    0.8442|
TUD-Stadtmitte      |    0.3568|   0.43991|   0.29182|   0.54371|   0.41616|   0.78402|
COMBINED            |   0.38358|   0.39764|   0.37772|   0.51055|   0.37057|   0.86534|
```

## Results

### Detections

#### Original detection scores (from FileDetectionsProvider)

Command to run the sequence in the given configuration

```bash
deep-sort run ./data/sequences -e F1 Precision Recall 
```

During the evaluation there are a noticeable amount of misdetections.

![FileDetectionProvider results](./resources/file_detections_provider.png)


```text
                    |F1        |Precision |Recall    |
KITTI-17            |   0.52665|   0.88362|   0.37511|
MOT16-09            |   0.41727|   0.94404|   0.26783|
MOT16-11            |    0.4623|   0.98828|   0.30172|
PETS09-S2L1         |   0.57651|   0.74787|   0.46904|
TUD-Campus          |   0.53687|    0.8442|   0.39358|
TUD-Stadtmitte      |   0.54371|   0.78402|   0.41616|
COMBINED            |   0.51055|   0.86534|   0.37057|
```

## Acknowledgement

Project is based on a [DeepSort algorithm implementation](https://github.com/nwojke/deep_sort) originally proposed in ["Simple Online and Realtime Tracking with a Deep Association Metric"](https://arxiv.org/abs/1703.07402).
The original project is licensed under Gnu General Public License.
