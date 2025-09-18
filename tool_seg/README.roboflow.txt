
Tools segmentation - v1 2025-08-12 5:40pm
==============================

This dataset was exported via roboflow.com on August 12, 2025 at 8:43 AM GMT

Roboflow is an end-to-end computer vision platform that helps you
* collaborate with your team on computer vision projects
* collect & organize images
* understand and search unstructured image data
* annotate, and create datasets
* export, train, and deploy computer vision models
* use active learning to improve your dataset over time

For state of the art Computer Vision training notebooks you can use with this dataset,
visit https://github.com/roboflow/notebooks

To find over 100k other datasets and pre-trained models, visit https://universe.roboflow.com

The dataset includes 1578 images.
Search-for-tools are annotated in YOLOv11 format.

The following pre-processing was applied to each image:
* Resize to 640x640 (Stretch)
* Auto-contrast via adaptive equalization

The following augmentation was applied to create 3 versions of each source image:
* Equal probability of one of the following 90-degree rotations: none, clockwise, counter-clockwise
* Random rotation of between -15 and +15 degrees
* Random Gaussian blur of between 0 and 1.8 pixels


