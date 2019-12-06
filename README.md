# Keyframe-Proposal-Network
1. Data preparation
- In this project, we used two popular public dataset:JHMDB and Penn Action
- Please download JHMDB from: http://jhmdb.is.tue.mpg.de/
- please download Penn Action: https://dreamdragon.github.io/PennAction/
- For those two dataset, bounding box for a person will be used in this project; Penn Action directly provides the person bounding box, but JHMDB doesn't. So, we refered
https://github.com/lawy623/LSTM_Pose_Machines/blob/master/dataset/JHMDB/utils/getBox.m to get bounding box, and saved .mat file for each of video
- We pre-trained dictionary of human skeleton by using https://github.com/liuem607/DYAN

2. Pre-requirement
- Python 3.6
- Pytorch > 0.4.0
- CUDA > 9.0
- Device for running time: NVIDIA GTX 1080 ti
- Please install package from https://github.com/sovrasov/flops-counter.pytorch to compute flops

3. Getting Start
- modify all path for data or models
- For training: please run 'main_*.py'
- For validation: please run 'test_*.py'
- For testing: we used https://github.com/microsoft/human-pose-estimation.pytorch with model which trained on MPII to get pose estimations on our key frames; after you get estimations from Simple Baseline, please replace skeleton information in the dataloader for test split

4. Evaluation
- please refer 'eval_PCKh.py' to get evaluation code
- to compute 

5. Download pre-trained model
- Create a folder named 'models'
- Download our pre-trained model and add them to 'models'

6. License
- The use of this software is RESTRICTED to non-commercial research and educational purposes
