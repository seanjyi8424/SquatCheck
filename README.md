# SquatCheck-UCR

![VID_DEMO](https://github.com/seanjyi8424/SquatCheckAI-UCR/assets/108261874/d40a877c-fd64-4492-9a4d-47acc16497a5)


## What is it?

A Machine Learning model that uses the OpenPose API and keypoint data as input to identify whether the exercise being performed is a valid squat. The original implementation utilized training off a personal computer and then executed on a Jetson Nano 4GB. Final project from my Edge Computing class.

## Task Distribution Diagram
![image](https://github.com/seanjyi8424/SquatCheckAI-UCR/assets/108261874/442dfe6c-5652-4fa4-8023-69d9d679274f)

## Requirements
1x Webcam (for live implementation)

[OpenPose API](https://github.com/CMU-Perceptual-Computing-Lab/openpose) 

OpenPose Dependencies (included in repository)

Windows 11

Nvidia GPU with updated Drivers

Python 3.7

numpy & opencv: sudo pip install numpy opencv-python

Visual Studio Community 2019

CUDA 11.1 & cuDNN 8.1.0

CMake GUI
