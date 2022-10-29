# DNN_POSE
  Object 6d pose estimation using an RGB-D camera.

  Project under [Barton Research Group](https://brg.engin.umich.edu/), via [Multidisciplinary-Design-Program (MDP)](https://mdp.engin.umich.edu/)
  at University of Michigan (UMich).

# How to run
  1) Run DNN_POSE pose estimation
```
cd ./dnn_pose/src/yolact/
python3 pose_estimation.py --trained_model=your_weights.pth --config=yolact_plus_resnet50_config --score_threshold=0.90
```
