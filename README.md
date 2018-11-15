# cpn_tensorflow

Re-implement [Cascaded Pyramid Network for Multi-Person Pose Estimation](https://arxiv.org/abs/1711.07319) (CPN) in Tensorflow.

## Motivation

**The code is for FashionAI costume key point positioning global challenge**
- Name of the competition: **Tianchi FashionAI costume key point positioning global challenge**
- Team Name: **AILAB-ZJU**
- Rank of the first season: **70/2322**
- Rank of the second season: **41/2322**
- Best result: NE = 4.45% for the second season

## Abstract

The topic of multi-person pose estimation has been largely improved recently, especially with the development of convolutional neural network. However, there still exist a lot of challenging cases, such as occluded keypoints, invisible keypoints and complex background, which cannot be well addressed. In this paper, we present a novel network structure called Cascaded Pyramid Network (CPN) which targets to relieve the problem from these "hard" keypoints. More specifically, our algorithm includes two stages: GlobalNet and RefineNet. GlobalNet is a feature pyramid network which can successfully localize the "simple" keypoints like eyes and hands but may fail to precisely recognize the occluded or invisible keypoints. Our RefineNet tries explicitly handling the "hard" keypoints by integrating all levels of feature representations from the GlobalNet together with an online hard keypoint mining loss. In general, to address the multi-person pose estimation problem, a top-down pipeline is adopted to first generate a set of human bounding boxes based on a detector, followed by our CPN for keypoint localization in each human bounding box. Based on the proposed algorithm, we achieve state-of-art results on the COCO keypoint benchmark, with average precision at 73.0 on the COCO test-dev dataset and 72.1 on the COCO test-challenge dataset, which is a 19% relative improvement compared with 60.5 from the COCO 2016 keypoint challenge.Code (this https URL) and the detection results are publicly available for further research.

## Framwork

![framwork1](framwork1.png)

![framwork2](framwork2.png)

## Details

## Requrements

First, install tensorflow, then installsome python packages:

>pip install -r requirments.txt

## TODO

- [] Model conpression for fast pose estimation




