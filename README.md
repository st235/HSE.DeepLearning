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

_Please, do note, `--n_init 0` will be used during detections evaluation to initialise tracks as soon as possible._

#### FileDetectionsProvider [from original DeepSORT algorithm]

![FileDetectionsProvider](./resources/file_detections_provider.png)

_FileDetectionsProvider_ is the logic that **was used in the original DeepSORT** version of the project.
It reads the `det/det.txt` file and extract detections from it.

Command to run the sequence in the given configuration is:

```bash
deep-sort run ./data/sequences -e F1 Precision Recall --n_init 0
```

During the evaluation there is a noticeable amount of mis-detections.

![FileDetectionProvider results](./resources/detection_score_file_detections_provider_1.png)

**Final score**

```text
                    |F1        |Precision |Recall    |
KITTI-17            |   0.72068|    0.8506|   0.62518|
MOT16-09            |   0.54102|   0.93847|   0.38006|
MOT16-11            |   0.60966|   0.98326|   0.44179|
PETS09-S2L1         |    0.7887|   0.70876|   0.88896|
TUD-Campus          |   0.72296|   0.77215|   0.67967|
TUD-Stadtmitte      |   0.72562|   0.72719|   0.72405|
COMBINED            |   0.68477|   0.83007|   0.62329|
```

#### YoloV5DetectionsProvider

![YoloV5DetectionsProvider](./resources/yolov5_detections_provider.png)

This _DetectionsProvider_ implements [Yolo V5](https://github.com/ultralytics/yolov5) object detection model.

This project implements `YoloV5` as a submodule.

Detector supports models:
- Nano, aka [YOLOv5n](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.pt)
- Small, aka [YOLOv5s](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt)
- Medium, aka [YOLOv5m](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5m.pt)
- Large, aka [YOLOv5l](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5l.pt)
- Nano6, aka [YOLOv5n6](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n6.pt)

All models are available in [`data/yolov5_binaries`](./dependencies/yolov5_binaries). However, if you were not able
to locate some checkpoints, please, do check the links above or the YoloV5 official repository.

**YoloV5N**

Command to run the sequence in the given configuration is:

```bash
deep-sort run ./data/sequences -e F1 Precision Recall -d yolov5n --n_init 0
```

A lot of people were not recognized correctly.

![YoloV5 Nano](./resources/detection_score_yolov5n.png)

FPS score has also dropped dramatically across the whole dataset (almost twice). Though, please, approach this and
any further notices about FPS drops with a reasonable pragmatism: **the local setup which is used for this project
is CPU-bounded as does not support CUDA**.

**Final scores**

```text
                    |F1        |Precision |Recall    |
KITTI-17            |   0.34803|   0.83799|   0.21962|
MOT16-09            |   0.35138|   0.96578|   0.21476|
MOT16-11            |    0.4939|   0.97401|   0.33083|
PETS09-S2L1         |   0.61212|   0.94028|   0.45375|
TUD-Campus          |   0.57418|   0.93125|   0.41504|
TUD-Stadtmitte      |   0.66321|   0.99139|   0.49827|
COMBINED            |   0.50714|   0.94012|   0.35538|
```

**YoloV5N6**

Command to run the sequence in the given configuration is:

```bash
deep-sort run ./data/sequences -e F1 Precision Recall -d yolov5n6 --n_init 0
```

Seems like this implementation provides more detections.

![YoloV5 Nano 6](./resources/detection_score_yolov5n6.png)

Performs really badly on [`KITTI17`](./data/sequences/KITTI-17)

![YoloV5 Nano 6 KITTI17](./resources/detection_score_yolov5n6_kitti17.png)

**Final scores**

```text
                    |F1        |Precision |Recall    |
KITTI-17            |   0.38761|   0.89418|   0.24744|
MOT16-09            |   0.34508|   0.97522|   0.20963|
MOT16-11            |   0.48004|   0.96793|   0.31916|
PETS09-S2L1         |   0.51717|   0.94104|   0.35657|
TUD-Campus          |   0.69272|   0.95588|   0.54318|
TUD-Stadtmitte      |   0.71792|    0.9954|   0.56142|
COMBINED            |   0.52342|   0.95494|    0.3729|
```

**YoloV5S**

Command to run the sequence in the given configuration is:

```bash
deep-sort run ./data/sequences -e F1 Precision Recall -d yolov5s --n_init 0
```

Much more accurate than `nano` models.

![YoloV5 Small](./resources/detection_score_yolov5s.png)

**Final scores**

```text
                    |F1        |Precision |Recall    |
KITTI-17            |   0.56972|   0.89097|   0.41874|
MOT16-09            |   0.48913|   0.96685|   0.32737|
MOT16-11            |   0.58584|   0.96238|   0.42108|
PETS09-S2L1         |   0.82626|   0.93609|    0.7395|
TUD-Campus          |    0.6831|   0.92823|   0.54039|
TUD-Stadtmitte      |   0.81384|   0.98765|   0.69204|
COMBINED            |   0.66131|   0.94536|   0.52319|
```

**YoloV5M**

Command to run the sequence in the given configuration is:

```bash
deep-sort run ./data/sequences -e F1 Precision Recall -d yolov5m --n_init 0
```

Good quality of detections.

![YoloV5 Medium](./resources/detection_score_yolov5m.png)

Moreover, on **CPU-bounded** devices there are about ~4-5 FPS,
which should be better on devices with available GPU and CUDA processing.

**Final scores**

```text
                    |F1        |Precision |Recall    |
KITTI-17            |   0.63188|   0.89757|   0.48755|
MOT16-09            |   0.58421|   0.95429|   0.42096|
MOT16-11            |   0.65607|   0.95539|   0.49956|
PETS09-S2L1         |   0.87007|   0.92026|   0.82507|
TUD-Campus          |   0.75042|   0.94118|   0.62396|
TUD-Stadtmitte      |   0.84325|   0.98837|   0.73529|
COMBINED            |   0.72265|   0.94284|   0.59873|
```

❗️ It seems that this model _outperforms_ the original algorithm by **all metrics**.

**YoloV5L**

Command to run the sequence in the given configuration is:

```bash
deep-sort run ./data/sequences -e F1 Precision Recall -d yolov5l --n_init 0
```

Outperforms all other `YOLO` models. However, not applicable in runtime
**at least on CPU-bounded** devices. Yet, some techniques as resizing or
buffering can help to improve performance if this model is required for usage.

![YoloV5 Large](./resources/detection_score_yolov5l.png)

**Final scores**

```text
                    |F1        |Precision |Recall    |
KITTI-17            |    0.6747|   0.91919|   0.53294|
MOT16-09            |   0.64809|   0.94463|   0.49325|
MOT16-11            |   0.68928|   0.95127|   0.54044|
PETS09-S2L1         |   0.88497|    0.9073|   0.86372|
TUD-Campus          |   0.75208|   0.93388|   0.62953|
TUD-Stadtmitte      |   0.85617|   0.98101|   0.75952|
COMBINED            |   0.75088|   0.93955|   0.63657|
```

#### NanodetDetectionsProvider

![NanodetDetectionsProvider](./resources/nanodet_detections_provider.png)

[NanodetDetectionsProvider](src/deep_sort/detector/nanodet_detections_provider.py)
is using [Nanodet](https://github.com/RangiLyu/nanodet) model.

⚠️⚠️⚠️ Unfortunately, this library does not work on MacOS, therefore `Nanodet` used in this project is a _patched_ version.

This is the patch that was sent to the authors of `Nanodet`:
- [#516 Guard CUDA calls with an explicit check](https://github.com/RangiLyu/nanodet/pull/516)

_NanodetDetectionsProvider_ supports the following checkpoints:
- Legacy M
- PlusM320
- PlusM15X320
- PlusM416
- PlusM15X416 

**Legacy M**

Command to run the sequence in the given configuration is:

```bash
deep-sort run ./data/sequences -e F1 Precision Recall -d nanodet_legacy --n_init 0
```

Runs bad on: `PETS09-S2L1` and `MOT16-11` almost not detecting anything

![Nanodet Legacy](./resources/detection_score_nanodet_legacy.png)

**Final scores**

```text
                    |F1        |Precision |Recall    |
KITTI-17            |       0.0|       1.0|       0.0|
MOT16-09            |  0.047177|       1.0|  0.024158|
MOT16-11            |  0.021341|   0.95192|  0.010791|
PETS09-S2L1         |  0.025127|   0.93443|  0.012735|
TUD-Campus          |     0.205|       1.0|   0.11421|
TUD-Stadtmitte      |  0.055509|       1.0|  0.028547|
COMBINED            |  0.059026|   0.98106|   0.03174|
```

**PlusM320**

Command to run the sequence in the given configuration is:

```bash
deep-sort run ./data/sequences -e F1 Precision Recall -d nanodet_plusm320 --n_init 0
```

Still runs terrible on `PETS09-S2L1`

![Nanodet Plus M 320](./resources/detection_score_nanodet_plusm320.png)

**Final scores**

```text
                    |F1        |Precision |Recall    |
KITTI-17            |       0.0|       1.0|       0.0|
MOT16-09            |   0.03329|   0.98889|   0.01693|
MOT16-11            |  0.076786|   0.95325|  0.040004|
PETS09-S2L1         | 0.0013396|       1.0|0.00067024|
TUD-Campus          |   0.19598|       1.0|   0.10864|
TUD-Stadtmitte      |    0.1563|       1.0|  0.084775|
COMBINED            |  0.077283|   0.99036|  0.041836|
```

**PlusM1.5X320**

Command to run the sequence in the given configuration is:

```bash
deep-sort run ./data/sequences -e F1 Precision Recall -d nanodet_plusm15x320 --n_init 0
```

Performs better `PETS09-S2L1` but still far away from idea. Perhaps,
the objects in the frame are too small for this model.

![Nanodet Plus M 1.5x 320](./resources/detection_score_nanodet_plusm15x320.png)

**Final scores**

```text
                    |F1        |Precision |Recall    |
KITTI-17            | 0.0029197|       0.5| 0.0014641|
MOT16-09            |  0.062581|   0.96591|  0.032338|
MOT16-11            |    0.1272|   0.93572|  0.068236|
PETS09-S2L1         | 0.0080107|       1.0| 0.0040214|
TUD-Campus          |   0.30189|   0.98462|   0.17827|
TUD-Stadtmitte      |   0.11111|       1.0|  0.058824|
COMBINED            |   0.10228|   0.89771|  0.057193|
```

**PlusM416**

Command to run the sequence in the given configuration is:

```bash
deep-sort run ./data/sequences -e F1 Precision Recall -d nanodet_plusm416 --n_init 0
```

Cannot detect small objects

| Small objects                                                               | Big objects                                                                |
|-----------------------------------------------------------------------------|----------------------------------------------------------------------------|
| ![Nanodet Small ojects](./resources/detection_score_nanodet_plusm416_1.png) | ![Nanodet Big objects](./resources/detection_score_nanodet_plusm416_2.png) |

**Final scores**

```text
                    |F1        |Precision |Recall    |
KITTI-17            |       0.0|       1.0|       0.0|
MOT16-09            |  0.082771|   0.99561|  0.043181|
MOT16-11            |   0.15543|   0.92951|  0.084805|
PETS09-S2L1         | 0.0071238|       1.0| 0.0035746|
TUD-Campus          |   0.26087|   0.98182|   0.15042|
TUD-Stadtmitte      |    0.2445|       1.0|   0.13927|
COMBINED            |   0.12511|   0.98449|  0.070209|
```

**PlusM1.5X416**

Command to run the sequence in the given configuration is:

```bash
deep-sort run ./data/sequences -e F1 Precision Recall -d nanodet_plusm15x416 --n_init 0
```

The best performance across all `nanodets`. However, the final quality is
still not enough. Though it is worth mentioning that frame rate is much
better than using `Yolo`.

![Nanodet M 1.5X 416](./resources/detection_score_nanodet_plusm15x416.png)

**Final scores**

```text
                    |F1        |Precision |Recall    |
KITTI-17            | 0.0058394|       1.0| 0.0029283|
MOT16-09            |   0.10261|   0.95638|  0.054213|
MOT16-11            |   0.21776|   0.95752|   0.12285|
PETS09-S2L1         |  0.015954|   0.97297| 0.0080429|
TUD-Campus          |   0.31308|   0.97101|   0.18663|
TUD-Stadtmitte      |   0.36723|       1.0|   0.22491|
COMBINED            |   0.17041|   0.97631|  0.099929|
```

#### GroundTruthDetectionsProvider

![GroundTruthDetectionsProvider](./resources/ground_truth_detections_provider.png)

_GroundTruthDetectionsProvider_ returns ground truth as detections. It helps to check tracking metrics,
and should result 1.0 for detections evaluation.

⚠️ Though we are evaluating detections, **tracking** is still running and slightly affect
the final score as detections appears not immediately. 
It takes a little time (i.e. a few frames) from tracks to initialise.

```bash
deep-sort run ./data/sequences -e F1 Precision Recall -d gt --n_init 0
```

**Final scores**

```text
                    |F1        |Precision |Recall    |
KITTI-17            |   0.98827|   0.98972|   0.98682|
MOT16-09            |   0.99581|   0.99676|   0.99486|
MOT16-11            |    0.9934|   0.99432|   0.99248|
PETS09-S2L1         |    0.9962|   0.99687|   0.99553|
TUD-Campus          |   0.98319|   0.98873|   0.97772|
TUD-Stadtmitte      |    0.9883|   0.99044|   0.98616|
COMBINED            |   0.99086|   0.99281|   0.98893|
```

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

## Acknowledgement

Project is based on a [DeepSort algorithm implementation](https://github.com/nwojke/deep_sort) originally proposed in ["Simple Online and Realtime Tracking with a Deep Association Metric"](https://arxiv.org/abs/1703.07402).
The original project is licensed under Gnu General Public License.
